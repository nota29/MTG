import torch
from one_peace.models import from_pretrained
import os
import glob
import numpy as np

def get_audio_list(folder1, folder2):
    audio_list = []

    audio_list.extend(glob.glob(os.path.join(folder1, "*.flac")))
    audio_list.extend(glob.glob(os.path.join(folder2, "*.flac")))

    return audio_list

device = "cuda" if torch.cuda.is_available() else "cpu"

# "ONE-PEACE" can also be replaced with ckpt path
model = from_pretrained("one_peace/checkpoints/one-peace.pt", device=device, dtype="float32")


# process raw data
audio_list = get_audio_list("one_peace/dataset/fsd50K/audio/dev", "one_peace/dataset/fsd50K/audio/eval")  #the two datasets, dev, eval
print("Number of audio files:", len(audio_list))


for audio in audio_list:
    src_audios, audio_padding_masks = model.process_audio([audio])
    file_number = os.path.splitext(os.path.basename(audio))[0]
    

    with torch.no_grad():
        # extract normalized features
        audio_features = model.extract_audio_features(src_audios, audio_padding_masks)
      
    
    audio_features = audio_features.cpu().numpy() 
    folder = "embeddings"           #the output folder, with the embeddings
    filepath = os.path.join(folder, file_number)
    np.save(filepath, audio_features)
