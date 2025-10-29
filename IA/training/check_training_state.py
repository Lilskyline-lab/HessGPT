#check_training_state.py
#!/usr/bin/env python3
"""
Script rapide pour vérifier l'état d'entraînement (pour Colab)
Usage: python check_training_state.py
"""

import sys
import os
from pathlib import Path
import torch

# Chemin des checkpoints
CHECKPOINT_DIR = Path("./IA/saved_models/my_llm/checkpoints")

def check_training_state():
    """Vérifie et affiche l'état d'entraînement"""
    print("\n" + "="*70)
    print("📊 ÉTAT D'ENTRAÎNEMENT - VÉRIFICATION RAPIDE")
    print("="*70)
    
    # Chercher l'état d'entraînement
    state_path = CHECKPOINT_DIR / "training_state.pt"
    
    if not state_path.exists():
        # Chercher un backup
        backups = sorted(
            CHECKPOINT_DIR.glob("state_backup_*.pt"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        if backups:
            state_path = backups[0]
            print(f"⚠️  État principal introuvable, utilisation du backup: {state_path.name}")
        else:
            print("❌ Aucun état d'entraînement trouvé")
            print("💡 L'entraînement démarrera depuis le début")
            print("="*70 + "\n")
            return
    
    # Charger l'état
    try:
        state = torch.load(state_path, map_location='cpu')
    except:
        state = torch.load(state_path, map_location='cpu', weights_only=False)
    
    # Afficher les infos
    print(f"\n✅ État d'entraînement trouvé: {state_path.name}")
    print(f"\n📅 Dernière sauvegarde: {state.get('timestamp', 'inconnu')}")
    print(f"\n🔢 Progression:")
    print(f"   • Époque: {state.get('epoch', 0)}")
    print(f"   • Batch actuel: {state.get('batch_idx', 0)}")
    print(f"   • Total batches traités: {state.get('total_batches_processed', 0):,}")
    print(f"\n📊 Métriques:")
    print(f"   • Train Loss: {state.get('train_loss', 0):.4f}")
    print(f"   • Val Loss: {state.get('val_loss', 0):.4f}")
    print(f"   • Best Val Loss: {state.get('best_val_loss', float('inf')):.4f}")
    
    # Estimation de la progression
    total_batches = state.get('total_batches_processed', 0)
    if total_batches > 0:
        print(f"\n📈 Estimation:")
        # En moyenne 10-20k batches par dataset de 30k exemples
        estimated_examples = total_batches * 8  # batch_size = 8
        print(f"   • Exemples déjà vus: ~{estimated_examples:,}")
        print(f"   • Prochaine sauvegarde dans: {1000 - (state.get('batch_idx', 0) % 1000)} batches")
    
    print(f"\n💡 Commande pour reprendre:")
    print(f"   python LoRAFineTuning.py")
    print(f"   → L'entraînement reprendra automatiquement au batch {state.get('batch_idx', 0) + 1}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    check_training_state()
