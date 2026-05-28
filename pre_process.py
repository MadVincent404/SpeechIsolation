import os
import random
from pathlib import Path
import torch
import torchaudio

#CONFIGURATION DES CHEMINS
FR_SPEECH_DIR = Path("To_Define")
EN_SPEECH_DIR = Path("To_Define")
NOISE_DIR = Path("To_Define")
OUTPUT_DIR = Path("To_Define")

#Paramètres
NUM_SAMPLES = 20
TARGET_SR = 16000
DURATION_SEC = 4 
TARGET_LEN = TARGET_SR * DURATION_SEC

#FONCTIONS UTILITAIRES

def get_all_audio_files(directory, extensions=[".wav", ".flac", ".mp3"]):
    """Récupère récursivement tous les fichiers audio d'un dossier."""
    audio_files = []
    for ext in extensions:
        audio_files.extend(list(directory.rglob(f"*{ext}")))
        audio_files.extend(list(directory.rglob(f"*{ext.upper()}")))
    return audio_files

def load_and_resample(file_path, target_sr=16000):
    """Charge un audio, le convertit en mono et le rééchantillonne si nécessaire."""
    waveform, sr = torchaudio.load(file_path)
    #Convertir en mono si stéréo
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    # Rééchantillonnage
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

def adjust_length(waveform, target_len):
    """Ajuste la taille de l'audio (pad si trop court, crop aléatoire si trop long)."""
    current_len = waveform.size(1)
    if current_len == target_len:
        return waveform
    elif current_len < target_len:
        # Padding avec du silence (ou vous pouvez répéter le signal)
        pad_len = target_len - current_len
        return torch.nn.functional.pad(waveform, (0, pad_len))
    else:
        # Crop aléatoire
        start = random.randint(0, current_len - target_len)
        return waveform[:, start:start + target_len]

def mix_signals(speech, noise, snr_db=5):
    """Mélange la voix et le bruit avec un SNR (Rapport Signal/Bruit) spécifique."""
    speech_power = speech.pow(2).mean()
    noise_power = noise.pow(2).mean()
    
    #Éviter la division par zéro
    if noise_power == 0:
        return speech + noise
        
    #Cacul du facteur d'échelle pour le bruit selon le SNR choisi
    snr = 10 ** (snr_db / 10.0)
    scale = torch.sqrt(speech_power / (noise_power * snr))
    return speech + scale * noise

# PROCESSUS PRINCIPAL

def generate_dataset():
    # 1. Collecte des fichiers
    print("Indexation des fichiers audio...")
    fr_files = get_all_audio_files(FR_SPEECH_DIR)
    en_files = get_all_audio_files(EN_SPEECH_DIR)
    noise_files = get_all_audio_files(NOISE_DIR)
    
    if not fr_files or not en_files or not noise_files:
        raise ValueError("Erreur : Un des dossiers sources est vide ou introuvable.")

    print(f"Trouvé : {len(fr_files)} audios FR, {len(en_files)} audios EN, {len(noise_files)} bruits.")

    #Configuration des couples Langue/Fichiers
    languages = {
        "french": fr_files,
        "english": en_files
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for lang, files in languages.items():
        print(f"\nGénération des échantillons pour : {lang}...")
        
        for i in range(1, NUM_SAMPLES + 1):
            speech_main_path, speech_interf_path = random.sample(files, 2)
            noise_path = random.choice(noise_files)
            
            # Chargement et normalisation de la durée/sample rate
            speech_main = adjust_length(load_and_resample(speech_main_path, TARGET_SR), TARGET_LEN)
            speech_interf = adjust_length(load_and_resample(speech_interf_path, TARGET_SR), TARGET_LEN)
            noise = adjust_length(load_and_resample(noise_path, TARGET_SR), TARGET_LEN)
            
            speech_with_noise = mix_signals(speech_main, noise, snr_db=5)
            
            # mix
            speech_mix = mix_signals(speech_main, speech_interf, snr_db=3)
            speech_mix = mix_signals(speech_mix, noise, snr_db=5)
            
            # Normalisation
            max_val = max(speech_mix.abs().max(), speech_with_noise.abs().max(), 1.0)
            
            sample_prefix = OUTPUT_DIR / f"{lang}_sample_{i:02d}"
            
            torchaudio.save(f"{sample_prefix}_speech_original.wav", speech_main / max_val, TARGET_SR)
            torchaudio.save(f"{sample_prefix}_noise_original.wav", noise / max_val, TARGET_SR)
            torchaudio.save(f"{sample_prefix}_speech_with_noise.wav", speech_with_noise / max_val, TARGET_SR)
            torchaudio.save(f"{sample_prefix}_speech_mix.wav", speech_mix / max_val, TARGET_SR)
            torchaudio.save(f"{sample_prefix}_speech_interferent.wav", speech_interf / max_val, TARGET_SR)

    print(f"\nTerminé ! Tous les fichiers sont dans le dossier : '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    generate_dataset()