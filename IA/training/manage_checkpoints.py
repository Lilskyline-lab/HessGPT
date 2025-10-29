
#!/usr/bin/env python3
"""
Script de gestion des checkpoints d'entraînement
Usage:
    python manage_checkpoints.py --list                    # Liste les checkpoints
    python manage_checkpoints.py --info <checkpoint.pt>    # Info sur un checkpoint
    python manage_checkpoints.py --clean                   # Nettoie les vieux checkpoints
    python manage_checkpoints.py --best                    # Info sur le meilleur checkpoint
"""

import argparse
import sys
import os
from pathlib import Path
import torch
import json

# Ajouter le chemin du projet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CHECKPOINT_DIR = "./IA/saved_models/my_llm/checkpoints"

def list_checkpoints(checkpoint_dir: str):
    """Liste tous les checkpoints"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Répertoire introuvable: {checkpoint_dir}")
        return
    
    checkpoints = sorted(
        checkpoint_path.glob("checkpoint_*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )
    
    if not checkpoints:
        print("❌ Aucun checkpoint trouvé")
        return
    
    print("\n" + "="*70)
    print("📋 CHECKPOINTS DISPONIBLES")
    print("="*70)
    
    for i, cp in enumerate(checkpoints):
        try:
            data = torch.load(cp, map_location='cpu')
        except:
            data = torch.load(cp, map_location='cpu', weights_only=False)
        
        size_mb = cp.stat().st_size / (1024 * 1024)
        
        print(f"\n{i+1}. {cp.name}")
        print(f"   📁 Taille: {size_mb:.2f} MB")
        print(f"   📅 Date: {data.get('timestamp', 'unknown')}")
        print(f"   🔢 Époque: {data.get('epoch', 0)}, Batch: {data.get('batch_idx', 0)}")
        print(f"   📊 Train Loss: {data.get('train_loss', 0):.4f}")
        print(f"   📉 Val Loss: {data.get('val_loss', 0):.4f}")
    
    print("\n" + "="*70)
    print(f"Total: {len(checkpoints)} checkpoints")
    
    # Taille totale
    total_size = sum(cp.stat().st_size for cp in checkpoints) / (1024 * 1024)
    print(f"Espace utilisé: {total_size:.2f} MB")
    print("="*70 + "\n")

def checkpoint_info(checkpoint_path: str):
    """Affiche les détails d'un checkpoint"""
    cp_path = Path(checkpoint_path)
    
    if not cp_path.exists():
        print(f"❌ Checkpoint introuvable: {checkpoint_path}")
        return
    
    try:
        data = torch.load(cp_path, map_location='cpu')
    except:
        data = torch.load(cp_path, map_location='cpu', weights_only=False)
    
    print("\n" + "="*70)
    print(f"📋 DÉTAILS DU CHECKPOINT: {cp_path.name}")
    print("="*70)
    
    print(f"\n📅 Métadonnées:")
    print(f"   Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"   Device: {data.get('device', 'unknown')}")
    
    print(f"\n🔢 Progression:")
    print(f"   Époque: {data.get('epoch', 0)}")
    print(f"   Batch: {data.get('batch_idx', 0)}")
    print(f"   Global Step: {data.get('global_step', 0)}")
    
    print(f"\n📊 Métriques:")
    print(f"   Train Loss: {data.get('train_loss', 0):.4f}")
    print(f"   Val Loss: {data.get('val_loss', 0):.4f}")
    print(f"   Best Val Loss: {data.get('best_val_loss', 0):.4f}")
    
    print(f"\n🔧 Configuration LoRA:")
    lora_cfg = data.get('lora_config', {})
    print(f"   Rank: {lora_cfg.get('rank', 'N/A')}")
    print(f"   Alpha: {lora_cfg.get('alpha', 'N/A')}")
    print(f"   Dropout: {lora_cfg.get('dropout', 'N/A')}")
    
    print(f"\n📚 Configuration Modèle:")
    model_cfg = data.get('model_config', {})
    print(f"   Vocab Size: {model_cfg.get('vocab_size', 'N/A')}")
    print(f"   Embed Dim: {model_cfg.get('embed_dim', 'N/A')}")
    print(f"   Num Heads: {model_cfg.get('num_heads', 'N/A')}")
    print(f"   Num Layers: {model_cfg.get('num_layers', 'N/A')}")
    
    print(f"\n💾 Historique:")
    history = data.get('history', {})
    print(f"   Total exemples: {history.get('total_qa_trained', 0):,}")
    print(f"   Cycles complétés: {len(history.get('cycles', []))}")
    
    size_mb = cp_path.stat().st_size / (1024 * 1024)
    print(f"\n📁 Fichier:")
    print(f"   Taille: {size_mb:.2f} MB")
    print(f"   Chemin: {cp_path.absolute()}")
    
    print("="*70 + "\n")

def show_best_checkpoint(checkpoint_dir: str):
    """Affiche le meilleur checkpoint"""
    best_path = Path(checkpoint_dir) / "best_checkpoint.pt"
    
    if not best_path.exists():
        print("❌ Aucun 'best checkpoint' trouvé")
        return
    
    checkpoint_info(str(best_path))

def clean_old_checkpoints(checkpoint_dir: str, keep_last: int = 3):
    """Nettoie les vieux checkpoints"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        print(f"❌ Répertoire introuvable: {checkpoint_dir}")
        return
    
    checkpoints = sorted(
        [f for f in checkpoint_path.glob("checkpoint_epoch*.pt")],
        key=lambda x: x.stat().st_mtime
    )
    
    if len(checkpoints) <= keep_last:
        print(f"✅ Seulement {len(checkpoints)} checkpoints, rien à nettoyer")
        return
    
    to_delete = checkpoints[:-keep_last]
    total_size = sum(cp.stat().st_size for cp in to_delete) / (1024 * 1024)
    
    print(f"\n🗑️  NETTOYAGE DES CHECKPOINTS")
    print(f"Garder les {keep_last} derniers")
    print(f"Supprimer {len(to_delete)} checkpoints ({total_size:.2f} MB)")
    
    response = input("\nConfirmer? (o/n): ").lower().strip()
    
    if response in ['o', 'oui', 'y', 'yes']:
        for cp in to_delete:
            cp.unlink()
            print(f"   ✓ Supprimé: {cp.name}")
        print(f"\n✅ {len(to_delete)} checkpoints supprimés ({total_size:.2f} MB libérés)")
    else:
        print("❌ Annulé")

def main():
    parser = argparse.ArgumentParser(description="Gestion des checkpoints d'entraînement")
    parser.add_argument("--dir", default=DEFAULT_CHECKPOINT_DIR, help="Répertoire des checkpoints")
    parser.add_argument("--list", action="store_true", help="Liste tous les checkpoints")
    parser.add_argument("--info", type=str, help="Info détaillée sur un checkpoint")
    parser.add_argument("--best", action="store_true", help="Info sur le meilleur checkpoint")
    parser.add_argument("--clean", action="store_true", help="Nettoie les vieux checkpoints")
    parser.add_argument("--keep", type=int, default=3, help="Nombre de checkpoints à garder (défaut: 3)")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.dir)
    elif args.info:
        checkpoint_info(args.info)
    elif args.best:
        show_best_checkpoint(args.dir)
    elif args.clean:
        clean_old_checkpoints(args.dir, args.keep)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
