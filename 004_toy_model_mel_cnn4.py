import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from datasets import load_dataset
import wandb
from huggingface_hub import HfApi, HfFolder, Repository, create_repo, upload_folder, hf_hub_download
import os
import librosa
import numpy as np
import torchaudio

def random_time_shift(audio, shift_max=0.2):
    # shift_max is the max fraction of total length to shift
    shift = np.random.randint(int(len(audio) * shift_max))
    if np.random.rand() > 0.5:
        shift = -shift
    augmented = np.roll(audio, shift)
    return augmented

def random_time_shift_torch(audio, shift_max=0.03):
    shift = int(audio.shape[-1] * shift_max * np.random.uniform(-1, 1))
    return torch.roll(audio, shifts=shift, dims=-1)

def add_noise(audio, noise_level=0.005):
    noise = np.random.randn(len(audio))
    augmented = audio + noise_level * noise
    return augmented

def add_noise_torch(audio, noise_level=0.002):
    noise = torch.randn_like(audio)
    return audio + noise_level * noise

def random_pitch_shift(audio, sr, n_steps=2):
    # n_steps: max number of semitones to shift
    steps = np.random.uniform(-n_steps, n_steps)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=steps)

def random_time_stretch(audio, rate_range=(0.8, 1.2)):
    rate = np.random.uniform(*rate_range)
    return librosa.effects.time_stretch(audio, rate=rate)

# CNN Model for Audio Classification: Deeper Conv2d on log-mel spectrogram
class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, input_channels=1, n_mels=128, max_time=1501, dropout=0.2):
        super(AudioCNN, self).__init__()
        # Input: (batch, 1, n_mels, time) e.g., (batch, 1, 128, 1501)
        self.conv2d_1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool2d_1 = nn.MaxPool2d(2)
        
        self.conv2d_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2d_2 = nn.BatchNorm2d(64)
        self.pool2d_2 = nn.MaxPool2d(2)
        
        # New deeper layers
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2d_3 = nn.BatchNorm2d(128)
        self.pool2d_3 = nn.MaxPool2d(2)
        
        self.conv2d_4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn2d_4 = nn.BatchNorm2d(256)
        self.pool2d_4 = nn.MaxPool2d(2)
        
        self.dropout = nn.Dropout(dropout)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (batch, 1, n_mels, time)
        x = self.conv2d_1(x)
        x = self.bn2d_1(x)
        x = self.relu(x)
        x = self.pool2d_1(x)
        x = self.dropout(x)

        x = self.conv2d_2(x)
        x = self.bn2d_2(x)
        x = self.relu(x)
        x = self.pool2d_2(x)
        x = self.dropout(x)

        x = self.conv2d_3(x)
        x = self.bn2d_3(x)
        x = self.relu(x)
        x = self.pool2d_3(x)
        x = self.dropout(x)

        x = self.conv2d_4(x)
        x = self.bn2d_4(x)
        x = self.relu(x)
        x = self.pool2d_4(x)
        x = self.dropout(x)

        x = self.global_pool(x)  # (batch, 256, 1, 1)
        x = x.view(x.size(0), -1)    # (batch, 256)
        x = self.fc(x)   # (batch, num_classes)
        return x

class AudioClassDataset(Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        audio = self.hf_dataset[idx]["audio"]["array"]  # 1D numpy array
        label = self.hf_dataset[idx]["classID"]           # integer class
        # Convert to torch tensors
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        return audio_tensor, label_tensor

class MelSpectrogramDataset(Dataset):
    def __init__(self, hf_dataset, sample_rate=16000, n_fft=2048, hop_length=512, n_mels=128, train=False):
        self.hf_dataset = hf_dataset
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.train = train  # <--- Add this
        self.sample_rate = sample_rate

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        self.db_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        audio = self.hf_dataset[idx]["audio"]["array"]
        sample_rate = self.hf_dataset[idx]["audio"]["sampling_rate"]
        label = self.hf_dataset[idx]["classID"]

        # Convert to torch tensor first!
        audio = torch.tensor(audio, dtype=torch.float32)

        # Only apply augmentation if training
        if self.train:
            # Randomly choose an augmentation
            aug_choice = np.random.choice([
                'shift', 
                'noise', 
                'none'
                ])
            if aug_choice == 'shift':
                audio = random_time_shift_torch(audio)
            elif aug_choice == 'noise':
                audio = add_noise_torch(audio)

        # Add channel dimension for torchaudio (1, N)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        # # Compute Mel-spectrogram
        # mel_spec = librosa.feature.melspectrogram(
        #     y=audio,
        #     sr=sample_rate,
        #     n_fft=self.n_fft,
        #     hop_length=self.hop_length,
        #     n_mels=self.n_mels
        # )
        # mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Compute Mel-spectrogram and convert to dB
        mel_spec = self.mel_transform(audio)  # (1, n_mels, time)
        mel_spec_db = self.db_transform(mel_spec)  # (1, n_mels, time)

        # Normalize to [0, 1]
        mel_norm = (mel_spec_db + 80) / 80  # from [-80, 0] → [0, 1]

        # Remove channel dimension for consistency with your collate_fn
        mel_norm = mel_norm.squeeze(0)  # (n_mels, time)

        label_tensor = torch.tensor(label, dtype=torch.long)

        # # normalize mel data
        # mel_norm = (mel_spec_db + 80) / 80  # from [-80, 0] → [0, 1]

        # # Convert to torch tensors
        # melnorm_tensor = torch.tensor(mel_norm, dtype=torch.float32)
        # label_tensor = torch.tensor(label, dtype=torch.long)
        return mel_norm, label_tensor

# Training loop with total loss and accuracy for train and test
def evaluate(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_X.size(0)
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy

def collate_fn_leftpad(batch):
    audios, labels = zip(*batch)
    max_len = max([a.shape[0] for a in audios])
    padded_audios = []
    for a in audios:
        pad_len = max_len - a.shape[0]
        # Pad at the beginning (left)
        if pad_len > 0:
            a_padded = torch.cat([torch.zeros(pad_len, dtype=a.dtype), a])
        else:
            a_padded = a
        padded_audios.append(a_padded)
    audios_tensor = torch.stack(padded_audios)
    # Add channel dimension: (batch, 1, time)
    audios_tensor = audios_tensor.unsqueeze(1)
    labels_tensor = torch.stack(labels)
    return audios_tensor, labels_tensor

def collate_fn_pad_to_max(batch, max_len=768_000, pad_left=False):
    audios, labels = zip(*batch)
    padded_audios = []
    for a in audios:
        pad_size = max_len - a.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, dtype=a.dtype)
            if pad_left:
                a_padded = torch.cat([pad, a])  # left pad
            else:
                a_padded = torch.cat([a, pad])  # right pad (default)
        else:
            a_padded = a  # no truncation, but if a is longer, you may want to raise an error
        padded_audios.append(a_padded)
    audios_tensor = torch.stack(padded_audios)
    labels_tensor = torch.stack(labels)
    # Add channel dimension for Conv1d: (batch, 1, time)
    audios_tensor = audios_tensor.unsqueeze(1)
    return audios_tensor, labels_tensor

def collate_fn_pad_mel_to_max(batch, max_len=1501, pad_left=False):
    mels, labels = zip(*batch)
    padded_mels = []
    for mel in mels:
        time_dim = mel.shape[1]
        pad_size = max_len - time_dim
        if pad_size > 0:
            # Pad along the time axis (axis=1)
            if pad_left:
                padded = F.pad(mel, (pad_size, 0), mode='constant', value=0)
            else:
                padded = F.pad(mel, (0, pad_size), mode='constant', value=0)
        else:
            padded = mel[:, :max_len]  # truncate if longer than max_len
        padded_mels.append(padded)
    mels_tensor = torch.stack(padded_mels)  # (batch, n_mels, max_len)
    mels_tensor = mels_tensor.unsqueeze(1)  # (batch, 1, n_mels, max_len) for Conv2d
    labels_tensor = torch.stack(labels)
    return mels_tensor, labels_tensor

def save_model_locally(model, save_dir="./trained_model", f_name='audio_cnn_model2.pt'):
    os.makedirs(save_dir, exist_ok=True)
    # Save decoder model and tokenizer
    torch.save(model.state_dict(), os.path.join(save_dir, f_name))
    print(f"Model saved locally to {save_dir}")

def push_model_to_hf(save_dir="./trained_model", repo_id="hiki-t/enc_dec_audio", path_in_repo=None):
    # Create repo if not exist
    api = HfApi()
    try:
        api.create_repo(repo_id, exist_ok=True)
    except Exception as e:
        print(f"Repo creation error (may already exist): {e}")
    # Upload folder
    upload_folder(
        repo_id=repo_id,
        folder_path=save_dir,
        path_in_repo=path_in_repo,
        commit_message=f"Upload encoder-decoder model weights and projections from {save_dir} to {path_in_repo if path_in_repo else '/'}",
        ignore_patterns=["*.tmp", "*.log"]
    )
    print(f"Model pushed to Hugging Face Hub: {repo_id} (folder: {save_dir} -> {path_in_repo if path_in_repo else '/'})")

def load_hf_weights(model, repo_id="hiki-t/enc_dec_audio", filename="audio_cnn_model.pt"):
    # Download the weights file from the repo
    weights_path = hf_hub_download(repo_id=repo_id, filename=filename)
    # Load the weights
    model.load_state_dict(torch.load(weights_path, weights_only=True))

def mixup_data(x, y, alpha=0.1):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# Compose SpecAugment transforms (define this ONCE, outside the loop)
spec_augment = torch.nn.Sequential(
    torchaudio.transforms.FrequencyMasking(freq_mask_param=3),
    torchaudio.transforms.TimeMasking(time_mask_param=3)
)

def calc_max_len(n_fft=2048, hop_length=512, n_mels=128, tmp_ds=None):
    n_fft=n_fft
    hop_length=hop_length
    n_mels=n_mels # or 64, or whatever you use

    # Use your training split
    max_len = 0
    for i in range(len(tmp_ds)):
        audio = tmp_ds[i]["audio"]["array"]
        sample_rate = tmp_ds[i]["audio"]["sampling_rate"]
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        if mel_spec.shape[1] > max_len:
            max_len = mel_spec.shape[1]

    print("Maximum mel spectrogram time length (max_len):", max_len)
    return max_len

# Example usage (replace with real data loading)
if __name__ == "__main__":
    # Dummy data: 100 samples, 1 channel, 16000 timesteps (e.g., 1 sec at 16kHz)
    num_samples = 100
    num_timesteps = 14004 # length of timestamps or t
    num_classes = 10
    image_width = 800
    batch_size = 64
    epochs = 10
    lr = 1e-4
    is_there_trained_weight = True
    save_model_file = "mel_cnn_model5.pt"
    save_dir = "./trained_model"
    hf_repo = "hiki-t/enc_dec_audio"

    ds = load_dataset("danavery/urbansound8K")

    # Split train into train+val+test
    split_ds = ds["train"].train_test_split(test_size=0.2, seed=42)
    train_ds = split_ds["train"]
    temp_ds = split_ds["test"].train_test_split(test_size=0.5, seed=42)
    val_ds = temp_ds["train"]
    test_ds = temp_ds["test"]

    # Get sample rate from the first sample
    sample_rate = train_ds[0]["audio"]["sampling_rate"]
    print("Sample rate:", sample_rate)

    # Optionally, repackage as a DatasetDict
    ds_splits = {"train": train_ds, "val": val_ds, "test": test_ds}

    n_fft = 2048
    hop_length = 512
    n_mels = 128
    num_workers = 8

    print('I am checking max len before organsing data')

    max_train_ds = 1501
    # max_train_ds = calc_max_len(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, tmp_ds=train_ds)
    # # max_val_ds = calc_max_len(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, tmp_ds=val_ds)
    # # max_test_ds = calc_max_len(n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, tmp_ds=test_ds)

    print('I am ready for data loader')

    # Create DataLoaders
    train_loader = DataLoader(
        MelSpectrogramDataset(train_ds, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, train=True), 
        batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, 
        collate_fn=lambda batch: collate_fn_pad_mel_to_max(batch, max_train_ds)
        )
    val_loader = DataLoader(
        MelSpectrogramDataset(val_ds, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, train=False), 
        batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, 
        collate_fn=lambda batch: collate_fn_pad_mel_to_max(batch, max_train_ds)
        )
    test_loader = DataLoader(
        MelSpectrogramDataset(test_ds, sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, train=False), 
        batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, 
        collate_fn=lambda batch: collate_fn_pad_mel_to_max(batch, max_train_ds)
        )

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AudioCNN(num_classes=num_classes, input_channels=1, n_mels=128, max_time=1501)
    if is_there_trained_weight:
        print("loading trained weights from huggingface")
        load_hf_weights(model, repo_id=hf_repo, filename=save_model_file)
    else:
        print("no trained weights, start from scratch")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # Initialize wandb
    wandb.init(
        project="enc_dec_from_scratch_audio", 
        entity="htsujimu-ucl", 
        config={
        "epochs": epochs,
        "batch_size": batch_size,
        "image_width": image_width,
        "max_len": 768_000,
        "learning_rate": lr,
    })
    wandb.watch(model, log="all")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        len_train_loader = len(train_loader)

        for batch_num, (batch_X, batch_y) in enumerate(train_loader):
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            # 1. Apply SpecAugment (per sample) # Increases sample diversity
            batch_X = spec_augment(batch_X)
            
            # # 2. Apply Mixup (per batch) # Mixes augmented samples
            # batch_X, targets_a, targets_b, lam = mixup_data(batch_X, batch_y, alpha=0.05)

            optimizer.zero_grad()
            outputs = model(batch_X)
            # loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_total += batch_X.size(0)

            # Log batch-wise metrics to wandb
            wandb.log({
                "batch_train_loss": loss.item(),
                "batch_train_acc": (preds == batch_y).float().mean().item(),
            })
            print(f"Batch {batch_num+1}/{len_train_loader}, Train Loss: {loss:.4f}")

        avg_train_loss = train_loss / train_total
        train_acc = train_correct / train_total

        # Validation (use val_loader instead of test_loader for validation metrics)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        # # In your training loop, after validation:
        # scheduler.step(val_loss)

        # Log epoch-wise metrics to wandb
        wandb.log({
            "epoch_train_loss": avg_train_loss,
            "epoch_train_acc": train_acc,
            "epoch_val_loss": val_loss,
            "epoch_val_acc": val_acc,
            "epoch": epoch + 1
        })
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        print("saving trained model weights")
        save_model_locally(model, save_dir=save_dir, f_name=save_model_file)
        push_model_to_hf(save_dir=save_dir, repo_id=hf_repo)

        # if epoch == 0:
        #     print("saving trained model weights")
        #     save_model_locally(model, save_dir=save_dir, f_name=save_model_file)
        #     push_model_to_hf(save_dir=save_dir, repo_id=hf_repo)
        #     best_val_loss = val_loss
        # elif val_loss < best_val_loss:
        #     print("saving trained model weights")
        #     save_model_locally(model, save_dir=save_dir, f_name=save_model_file)
        #     push_model_to_hf(save_dir=save_dir, repo_id=hf_repo)
        #     best_val_loss = val_loss
        # elif (best_val_loss - val_loss)/val_loss < accept_loss_up:
        #     print("loss increase is less than {accept_loss_up*100}%, let's save")
        #     save_model_locally(model, save_dir=save_dir, f_name=save_model_file)
        #     push_model_to_hf(save_dir=save_dir, repo_id=hf_repo)
        # else:
        #     print("performance is getting too worse! no saving")

    # Final test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion)
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc
    })
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    wandb.finish()

    print("Proto training complete. Replace dummy data with real audio features for actual use.")
