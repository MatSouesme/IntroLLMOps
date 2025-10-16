# ============================================================
# 1️⃣ INSTALLATION DES DÉPENDANCES (commande à exécuter dans ton terminal)
# ============================================================
# uv add peft transformers trl accelerate tensorboard torch
# (ajoute bitsandbytes si tu n'es PAS sur MacOS)
# uv add bitsandbytes
# ============================================================


# ============================================================
# 2️⃣ IMPORTS ET CONFIGURATION DE L'ENVIRONNEMENT
# ============================================================
import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig

# Vérifie si un GPU est dispo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Entraînement sur : {device.upper()}")


# ============================================================
# 3️⃣ DÉFINITION DES CONFIGURATIONS DE FINE-TUNING
# ============================================================

# --- LoRA Config ---
lora_config = LoraConfig(
    r=8,                     # Taille du goulot (rank)
    lora_alpha=16,           # Facteur de mise à l'échelle
    target_modules=["q_proj", "v_proj"],  # Couches ciblées
    lora_dropout=0.1,        # Dropout
    bias="none",             # Pas d'adaptation de biais
    task_type="CAUSAL_LM"    # Tâche : modélisation de langage
)
print("✅ LoRA Config créée :", lora_config)


# --- BitsAndBytes Config ---
# ⚠️ Saute cette section si tu es sur MacOS
try:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    print("✅ BitsAndBytes Config créée :", bnb_config)
except Exception as e:
    print("⚠️ Impossible d'importer BitsAndBytes (probablement sur MacOS).")
    bnb_config = None


# --- SFT Config (Supervised Fine-Tuning) ---
sft_config = SFTConfig(
    output_dir="./results",             # Dossier de sortie
    num_train_epochs=3,                 # Nombre d'époques
    per_device_train_batch_size=4,      # Batch par GPU
    gradient_accumulation_steps=4,      # Accumulation pour grand batch virtuel
    learning_rate=2e-4,                 # Learning rate
    logging_dir="./logs",               # Logs TensorBoard
    report_to=["tensorboard"],          # Visualisation TensorBoard
    save_strategy="epoch"               # Sauvegarde à chaque époque
)
print("✅ SFT Config créée :", sft_config)


# ============================================================
# 4️⃣ ISOLER LES HYPERPARAMÈTRES DANS UN DICTIONNAIRE
# ============================================================
hyperparams = {
    "model_name": "microsoft/Phi-3-mini-4k-instruct",
    "num_train_epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 4,
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.1,
    "bnb_4bit": True if bnb_config is not None else False
}
print("🔧 Hyperparamètres configurés :", hyperparams)


# ============================================================
# 5️⃣ CHARGEMENT DU MODÈLE ET PRÉPARATION POUR LORA
# ============================================================
model_name = hyperparams["model_name"]

# --- Chargement du tokenizer ---
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Sécurité pour CausalLM

# --- Chargement du modèle Phi-3 ---
# Cette section gère automatiquement les cas MacOS / CPU / GPU
if torch.cuda.is_available() and bnb_config is not None:
    print("⚙️ Chargement du modèle Phi-3 avec quantization 4-bit (GPU).")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
else:
    print("⚙️ Chargement du modèle Phi-3 sans quantization (CPU ou MacOS).")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=None,
        torch_dtype=torch.float32,
    )

print(f"✅ Modèle {model_name} chargé avec succès sur {device.upper()}.")

# --- Préparation pour le fine-tuning LoRA ---
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)



# ============================================================
# 6️⃣ VÉRIFICATION DES PARAMÈTRES ENTRAÎNABLES ET ADAPTATEURS LORA
# ============================================================

# Calcule le ratio de paramètres entraînables
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
ratio = 100 * trainable_params / total_params

print(f"🔍 Nombre de paramètres entraînables : {trainable_params:,} / {total_params:,}")
print(f"👉 Ratio : {ratio:.4f}% du modèle total")

# Vérifie les modules LoRA ajoutés
print("\n🔧 Adaptateurs LoRA ajoutés :")
for name, module in model.named_modules():
    if "lora" in name.lower():
        print(" -", name)

