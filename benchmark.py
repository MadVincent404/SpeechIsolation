import os
import glob
import torch
import torchaudio
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchaudio.transforms as T
from pesq import pesq
from pystoi import stoi

# Framework SpeechBrain
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.inference.enhancement import SpectralMaskEnhancement

# Framework Asteroid
from asteroid.models import ConvTasNet

# DÉFINITION DES MÉTRIQUES

def compute_si_sdr(ref, est):
    """Calcule le Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)"""
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    target_energy = np.sum(target ** 2)
    noise_energy = np.sum(noise ** 2)
    return 10 * np.log10(target_energy / (noise_energy + 1e-8) + 1e-8)

def compute_pesq_wb(ref, est, sr=16000):
    """Calcule le PESQ Wideband (16kHz)"""
    try:
        return pesq(sr, ref, est, 'wb')
    except Exception:
        return float('nan')

def compute_stoi_score(ref, est, sr=16000):
    """Calcule le score STOI"""
    try:
        return stoi(ref, est, sr, extended=False)
    except Exception:
        return float('nan')

def compute_l1_mel(ref, est, sr=16000):
    """Calcule la distance L1 entre les Log-Mel Spectrogrammes"""
    ref_t = torch.from_numpy(ref).float()
    est_t = torch.from_numpy(est).float()
    
    mel_transform = T.MelSpectrogram(
        sample_rate=sr, n_fft=400, hop_length=160, n_mels=80
    )
    
    ref_mel = torch.log(mel_transform(ref_t) + 1e-5)
    est_mel = torch.log(mel_transform(est_t) + 1e-5)
    
    return F.l1_loss(ref_mel, est_mel).item()

#NFIGURATION ET CHARGEMENT SÉCURISÉ DES MODÈLES

def main():
    # Correction du format de device pour éviter les alertes de parsing de SpeechBrain
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"--> Utilisation de l'appareil : {device}")

    # Dictionnaires pour stocker les modèles correctement chargés
    enhancement_models = {}
    separation_models = {}

    print("--> Chargement des modèles SpeechBrain...")
    
    # SB SepFormer Enhancement
    try:
        enhancement_models["SB-SepFormer-Enhancement"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham16k-enhancement",
            savedir="pretrained_models/sepformer_wham16k_enh",
            run_opts={"device": device}
        )
    except Exception as e:
        print(f" Impossible de charger SB-SepFormer-Enhancement : {e}")

    # SepFormer-DNS4
    try:
        enhancement_models["SB-SepFormer-DNS4"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-dns4-16k-enhancement",
            savedir="pretrained_models/sepformer_dns4",
            run_opts={"device": device}
        )
    except Exception as e:
        print(f" Impossible de charger SB-SepFormer-DNS4 : {e}")

    # SB SepFormer WHAMR
    try:
        separation_models["SB-SepFormer-WHAMR16k"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir="pretrained_models/sepformer16k_whamr",
            run_opts={"device": device}
        )
    except Exception as e:
        print(f" Impossible de charger SB-SepFormer-WHAMR16k : {e}")

    print("--> Chargement des modèles Asteroid...")
    
    # Asteroid Libri1Mix
    try:
        model_ast_enh = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k").to(device)
        model_ast_enh.eval()
        enhancement_models["Ast-ConvTasNet-Libri1Mix"] = model_ast_enh
    except Exception as e:
        print(f" Impossible de charger Ast-ConvTasNet-Libri1Mix : {e}")
    
    # Asteroid Libri2Mix
    try:
        model_ast_sep = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k").to(device)
        model_ast_sep.eval()
        separation_models["Ast-ConvTasNet-Libri2Mix"] = model_ast_sep
    except Exception as e:
        print(f" Impossible de charger Ast-ConvTasNet-Libri2Mix : {e}")

    languages = ["english", "french"]
    results = []

    #BOUCLE DE BENCHMARK
    for lang in languages:
        dir_path = f"samples/{lang}"
        if not os.path.exists(dir_path):
            print(f" Répertoire {dir_path} introuvable. Passage à la suite.")
            continue
        
        clean_files = glob.glob(os.path.join(dir_path, f"{lang}_sample_*_speech_original.wav"))
        sample_ids = sorted(list(set([os.path.basename(f).split("_")[2] for f in clean_files])))
        
        print(f"\nDébut du Benchmark : {lang.upper()} ({len(sample_ids)} fichiers") 

        for sample_id in sample_ids:
            # Référence propre
            ref_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_original.wav"
            ref_audio, sr = torchaudio.load(ref_path)
            if sr != 16000:
                ref_audio = torchaudio.functional.resample(ref_audio, sr, 16000)
                sr = 16000
            ref_np = ref_audio.mean(dim=0).numpy()

            # Speech Enhancement (1 voix + bruit)
            noise_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_with_noise.wav"
            if os.path.exists(noise_path):
                
                #Modèles d'enhancement enregistrés (SB + Asteroid)
                for model_name, model in enhancement_models.items():
                    try:
                        if model_name.startswith("SB"):
                            est_sources = model.separate_file(path=noise_path)
                            est_np = est_sources.squeeze(0).cpu().numpy()
                            if len(est_np.shape) > 1: est_np = est_np[:, 0]
                        else:  # Modèle Asteroid
                            mix_audio, _ = torchaudio.load(noise_path)
                            if mix_audio.shape[0] > 1: mix_audio = mix_audio.mean(dim=0, keepdim=True)
                            with torch.no_grad():
                                est_sources = model(mix_audio.to(device))
                            est_np = est_sources.squeeze(0).squeeze(0).cpu().numpy()

                        min_len = min(len(ref_np), len(est_np))
                        r, e = ref_np[:min_len], est_np[:min_len]
                        
                        results.append({
                            "Language": lang, "Sample_ID": sample_id, "Task": "Speech_With_Noise",
                            "Model": model_name, "PESQ": compute_pesq_wb(r, e, sr),
                            "STOI": compute_stoi_score(r, e, sr), "SI-SDR": compute_si_sdr(r, e), "L1_Mel": compute_l1_mel(r, e, sr)
                        })
                    except Exception as ex:
                        print(f"Erreur avec le modèle {model_name} sur le sample {sample_id}: {ex}")

            #Speech Separation (1 voix + 1 interférent + bruit)
            mix_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_mix.wav"
            if os.path.exists(mix_path):
                
                #Modèles de séparation enregistrés
                for model_name, model in separation_models.items():
                    try:
                        if model_name.startswith("SB"):
                            est_sources = model.separate_file(path=mix_path)
                            est_sources_np = est_sources.squeeze(0).cpu().numpy()
                        else:  # Modèle Asteroid
                            mix_audio, _ = torchaudio.load(mix_path)
                            if mix_audio.shape[0] > 1: mix_audio = mix_audio.mean(dim=0, keepdim=True)
                            with torch.no_grad():
                                est_sources = model(mix_audio.to(device))
                            est_sources_np = est_sources.squeeze(0).cpu().numpy() # [Sources, Time]

                        # Permutation Invariance Logic
                        best_sdr = -float('inf')
                        best_est = None
                        
                        if model_name.startswith("SB"):
                            num_channels = est_sources_np.shape[1] if len(est_sources_np.shape) > 1 else 1
                            if num_channels == 1:
                                best_est = est_sources_np
                            else:
                                for c in range(num_channels):
                                    est_c = est_sources_np[:, c]
                                    min_len = min(len(ref_np), len(est_c))
                                    sdr = compute_si_sdr(ref_np[:min_len], est_c[:min_len])
                                    if sdr > best_sdr:
                                        best_sdr = sdr
                                        best_est = est_c
                        else: # Asteroid
                            for s in range(est_sources_np.shape[0]):
                                est_s = est_sources_np[s, :]
                                min_len = min(len(ref_np), len(est_s))
                                sdr = compute_si_sdr(ref_np[:min_len], est_s[:min_len])
                                if sdr > best_sdr:
                                    best_sdr = sdr
                                    best_est = est_s

                        min_len = min(len(ref_np), len(best_est))
                        r, e = ref_np[:min_len], best_est[:min_len]
                        
                        results.append({
                            "Language": lang, "Sample_ID": sample_id, "Task": "Speech_Mix",
                            "Model": model_name, "PESQ": compute_pesq_wb(r, e, sr),
                            "STOI": compute_stoi_score(r, e, sr), "SI-SDR": compute_si_sdr(r, e), "L1_Mel": compute_l1_mel(r, e, sr)
                        })
                    except Exception as ex:
                        print(f"Erreur avec le modèle {model_name} sur le sample {sample_id}: {ex}")

    # AGRÉGATION ET AFFICHAGE DES RÉSULTATS
    if results:
        df = pd.DataFrame(results)
        df.to_csv("benchmark_detailed_results.csv", index=False)
        
        print("\n" + "="*80)
        print("RÉSULTATS SYNTHÉTIQUES COMPARES (MOYENNES)")
        print("="*80)
        summary = df.groupby(["Language", "Task", "Model"])[["PESQ", "STOI", "SI-SDR", "L1_Mel"]].mean()
        print(summary.to_string())
        print("="*80)
    else:
        print("\nAucune donnée n'a pu être traitée.")

if __name__ == "__main__":
    main()