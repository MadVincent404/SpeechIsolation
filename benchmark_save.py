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

from speechbrain.inference.separation import SepformerSeparation
from asteroid.models import ConvTasNet

#DÉFINITION DES MÉTRIQUES

def compute_si_sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(est, ref) / (np.dot(ref, ref) + 1e-8)
    target = alpha * ref
    noise = est - target
    return 10 * np.log10(np.sum(target ** 2) / (np.sum(noise ** 2) + 1e-8) + 1e-8)

def compute_pesq_wb(ref, est, sr=16000):
    try: return pesq(sr, ref, est, 'wb')
    except: return float('nan')

def compute_stoi_score(ref, est, sr=16000):
    try: return stoi(ref, est, sr, extended=False)
    except: return float('nan')

def compute_l1_mel(ref, est, sr=16000):
    ref_t = torch.from_numpy(ref).float()
    est_t = torch.from_numpy(est).float()
    mel_transform = T.MelSpectrogram(sample_rate=sr, n_fft=400, hop_length=160, n_mels=80)
    ref_mel = torch.log(mel_transform(ref_t) + 1e-5)
    est_mel = torch.log(mel_transform(est_t) + 1e-5)
    return F.l1_loss(ref_mel, est_mel).item()

#UTILITAIRE DE SAUVEGARDE AUDIO
def save_audio_helper(folder, filename, data_np, sr=16000):
    """Sauvegarde un tableau numpy au format WAV 16kHz Mono"""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    tensor = torch.from_numpy(data_np).unsqueeze(0).float() # [1, Time]
    torchaudio.save(path, tensor, sr)

#MAIN BENCHMARK & EXPORT
def main():
    MAX_SAMPLES = 5 
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"--> Utilisation de l'appareil : {device}")

    base_output_dir = "output_audio"
    enhancement_models = {}
    separation_models = {}

    print("--> Chargement des modèles...")
    try:
        enhancement_models["SB-SepFormer-Enhancement"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-wham16k-enhancement",
            savedir="pretrained_models/sepformer_wham16k_enh",
            run_opts={"device": device}
        )
    except Exception as e: print(f"Erreur SB-Enhancement: {e}")

    try:
        enhancement_models["SB-SepFormer-DNS4"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-dns4-16k-enhancement",
            savedir="pretrained_models/sepformer_dns4",
            run_opts={"device": device}
        )
    except Exception as e: print(f"Erreur SB-DNS4: {e}")

    try:
        separation_models["SB-SepFormer-WHAMR16k"] = SepformerSeparation.from_hparams(
            source="speechbrain/sepformer-whamr16k",
            savedir="pretrained_models/sepformer16k_whamr",
            run_opts={"device": device}
        )
    except Exception as e: print(f"Erreur SB-WHAMR: {e}")

    try:
        model_ast_enh = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri1Mix_enhsingle_16k").to(device)
        model_ast_enh.eval()
        enhancement_models["Ast-ConvTasNet-Libri1Mix"] = model_ast_enh
    except Exception as e: print(f"Erreur Asteroid Libri1Mix: {e}")
    
    try:
        model_ast_sep = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k").to(device)
        model_ast_sep.eval()
        separation_models["Ast-ConvTasNet-Libri2Mix"] = model_ast_sep
    except Exception as e: print(f"Erreur Asteroid Libri2Mix: {e}")

    languages = ["english", "french"]
    results = []

    for lang in languages:
        dir_path = f"samples/{lang}"
        if not os.path.exists(dir_path): continue
        
        clean_files = glob.glob(os.path.join(dir_path, f"{lang}_sample_*_speech_original.wav"))
        all_ids = sorted(list(set([os.path.basename(f).split("_")[2] for f in clean_files])))
        
        # Application du bridage aux 5 premiers samples trouvés
        sample_ids = all_ids[:MAX_SAMPLES]
        
        print(f"\n--- Traitement & Export : {lang.upper()} (Limité à {len(sample_ids)}/{len(all_ids)} samples) ---")

        for sample_id in sample_ids:
            sample_output_dir = os.path.join(base_output_dir, lang, f"sample_{sample_id}")
            
            # Référence propre
            ref_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_original.wav"
            ref_audio, sr = torchaudio.load(ref_path)
            ref_np = ref_audio.mean(dim=0).numpy()
            save_audio_helper(sample_output_dir, "01_ORIGINAL_clean.wav", ref_np)

            # Voix interferente
            interf_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_interferent.wav"
            if os.path.exists(interf_path):
                ai, _ = torchaudio.load(interf_path)
                save_audio_helper(sample_output_dir, "02_ORIGINAL_interferent.wav", ai.mean(dim=0).numpy())

            # Bruit original
            noise_orig_path = f"samples/{lang}/{lang}_sample_{sample_id}_noise_original.wav"
            if os.path.exists(noise_orig_path):
                no, _ = torchaudio.load(noise_orig_path)
                save_audio_helper(sample_output_dir, "03_ORIGINAL_noise.wav", no.mean(dim=0).numpy())

            # Entrée 1 : Speech with noise
            noise_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_with_noise.wav"
            if os.path.exists(noise_path):
                n_aud, _ = torchaudio.load(noise_path)
                n_np = n_aud.mean(dim=0).numpy()
                save_audio_helper(sample_output_dir, "04_INPUT_speech_with_noise.wav", n_np)
                
                for model_name, model in enhancement_models.items():
                    try:
                        if model_name.startswith("SB"):
                            est_sources = model.separate_file(path=noise_path)
                            est_np = est_sources.squeeze(0).cpu().numpy()
                            if len(est_np.shape) > 1: est_np = est_np[:, 0]
                        else:
                            with torch.no_grad():
                                est_sources = model(torch.from_numpy(n_np).unsqueeze(0).unsqueeze(0).to(device))
                            est_np = est_sources.squeeze(0).squeeze(0).cpu().numpy()

                        min_len = min(len(ref_np), len(est_np))
                        r, e = ref_np[:min_len], est_np[:min_len]
                        
                        save_audio_helper(sample_output_dir, f"OUTPUT_{model_name}.wav", e)

                        results.append({
                            "Language": lang, "Sample_ID": sample_id, "Task": "Speech_With_Noise",
                            "Model": model_name, "PESQ": compute_pesq_wb(r, e, sr),
                            "STOI": compute_stoi_score(r, e, sr), "SI-SDR": compute_si_sdr(r, e), "L1_Mel": compute_l1_mel(r, e, sr)
                        })
                    except Exception as ex: print(f"Erreur {model_name} sur {sample_id}: {ex}")

            # Entrée 2 : Speech Mix
            mix_path = f"samples/{lang}/{lang}_sample_{sample_id}_speech_mix.wav"
            if os.path.exists(mix_path):
                m_aud, _ = torchaudio.load(mix_path)
                m_np = m_aud.mean(dim=0).numpy()
                save_audio_helper(sample_output_dir, "05_INPUT_speech_mix.wav", m_np)
                
                for model_name, model in separation_models.items():
                    try:
                        if model_name.startswith("SB"):
                            est_sources = model.separate_file(path=mix_path)
                            est_sources_np = est_sources.squeeze(0).cpu().numpy()
                        else:
                            with torch.no_grad():
                                est_sources = model(torch.from_numpy(m_np).unsqueeze(0).unsqueeze(0).to(device))
                            est_sources_np = est_sources.squeeze(0).cpu().numpy()

                        best_sdr = -float('inf')
                        best_est = None
                        
                        if model_name.startswith("SB"):
                            num_channels = est_sources_np.shape[1] if len(est_sources_np.shape) > 1 else 1
                            if num_channels == 1: best_est = est_sources_np
                            else:
                                for c in range(num_channels):
                                    est_c = est_sources_np[:, c]
                                    min_len = min(len(ref_np), len(est_c))
                                    sdr = compute_si_sdr(ref_np[:min_len], est_c[:min_len])
                                    if sdr > best_sdr: best_sdr = sdr; best_est = est_c
                        else:
                            for s in range(est_sources_np.shape[0]):
                                est_s = est_sources_np[s, :]
                                min_len = min(len(ref_np), len(est_s))
                                sdr = compute_si_sdr(ref_np[:min_len], est_s[:min_len])
                                if sdr > best_sdr: best_sdr = sdr; best_est = est_s

                        min_len = min(len(ref_np), len(best_est))
                        r, e = ref_np[:min_len], best_est[:min_len]
                        
                        save_audio_helper(sample_output_dir, f"OUTPUT_{model_name}.wav", e)

                        results.append({
                            "Language": lang, "Sample_ID": sample_id, "Task": "Speech_Mix",
                            "Model": model_name, "PESQ": compute_pesq_wb(r, e, sr),
                            "STOI": compute_stoi_score(r, e, sr), "SI-SDR": compute_si_sdr(r, e), "L1_Mel": compute_l1_mel(r, e, sr)
                        })
                    except Exception as ex: print(f"Erreur {model_name} sur {sample_id}: {ex}")

    if results:
        df = pd.DataFrame(results)
        df.to_csv("benchmark_detailed_results.csv", index=False)
        print("\n" + "="*80)
        print("RÉSULTATS SYNTHÉTIQUES COMPARES (MOYENNES SUR 5 SAMPLES)")
        print("="*80)
        print(df.groupby(["Language", "Task", "Model"])[["PESQ", "STOI", "SI-SDR", "L1_Mel"]].mean().to_string())
        print("="*80)
        print(" -> Fichiers audio exportés dans './output_audio/' (5 par langue).")

if __name__ == "__main__":
    main()