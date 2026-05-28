import os


import argparse
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation as separator

def run_inference(input_audio_path, output_audio_path):
    """
    Charge le modèle SB-SepFormer-Enhancement, applique le traitement 
    sur le fichier bruité et sauvegarde le signal audio nettoyé.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Périphérique détecté : {device.upper()}")
    
    model_source = "speechbrain/sepformer-wham16k-enhancement"
    save_directory = "pretrained_models/sepformer-wham16k-enhancement"
    
    print("Chargement du modèle SepFormer...")
    model = separator.from_hparams(
        source=model_source,
        savedir=save_directory,
        run_opts={"device": device}
    )
    
    if not os.path.exists(input_audio_path):
        print(f"Erreur : Le fichier d'entrée '{input_audio_path}' n'existe pas.")
        return

    print(f"Traitement en cours de : {input_audio_path}")
    
    try:
        enhanced_speech = model.separate_file(path=input_audio_path)
        enhanced_speech = enhanced_speech.squeeze().cpu()
        
        if enhanced_speech.dim() == 1:
            enhanced_speech = enhanced_speech.unsqueeze(0)
            
        # Normalisation des chemins pour Windows (évite les conflits de slashes / et \)
        output_audio_path = os.path.normpath(output_audio_path)
        output_dir = os.path.dirname(output_audio_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        torchaudio.save(output_audio_path, enhanced_speech, sample_rate=16000)
        print(f"Amélioration réussie ! Fichier sauvegardé sous : {output_audio_path}")
        
    except Exception as e:
        print(f"Une erreur est survenue lors de l'inférence : {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline d'inférence SpeechBrain SepFormer")
    parser.add_argument("--input", type=str, required=True, help="Chemin vers le fichier audio bruité (.wav)")
    parser.add_argument("--output", type=str, default="output_enhanced.wav", help="Chemin du fichier de sortie")
    
    args = parser.parse_args()
    run_inference(args.input, args.output)