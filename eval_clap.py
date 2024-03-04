import laion_clap
from os.path import join as pjoin
import torch


def compute_average_score(clap, music_captions, audio_paths, batch_size=32):
    scores = []
    count = 0
    with torch.no_grad():
        while count < len(music_captions):
            batch_audio_files = audio_paths[count:count + batch_size]
            batch_captions = music_captions[count: count + batch_size]

            audio_emb = clap.get_audio_embedding_from_filelist(x=batch_audio_files, use_tensor=True)
            text_emb = clap.get_text_embedding(batch_captions, use_tensor=True)

            result = torch.sum(audio_emb * text_emb, dim=-1)
            print(f'{count}/{len(music_captions)}: {audio_emb.shape}, {text_emb.shape}, {result.shape}')
            scores.extend(list(result))

            count += batch_size
    return sum(scores) / len(scores)



if __name__ == "__main__":
    import pandas as pd

    # Ensure that number of audio files match number of captions
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--music_path",
        type=str,
        default=None,
        help="Where to load music",
    )
    opt = parser.parse_args()

    model = laion_clap.CLAP_Module(enable_fusion=False, device='cuda:2')
    model.load_ckpt()  # download the default pretrained checkpoint.
    # device = torch.device('cuda:1')
    # model = model.to(device)

    audio_path = opt.music_path
    audio_names = [filename for filename in os.listdir(audio_path) if filename.endswith(".wav") or filename.endswith(".mp3")]

    data = pd.read_csv('/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/My_Project/data/music/musiccaps-public.csv', index_col=0)

    column_as_list = []
    audio_paths = []
    gt_names = list(data.index)
    for pth in audio_names:
        filename = pth.split('.')[0]
        if filename in gt_names:
            audio_paths.append(f'{audio_path}/{pth}')
            column_as_list.append(data.loc[filename].caption)

    # Get all .wav files from the directory
    # audio_paths = [f"{audio_path}/{filename}" for filename in os.listdir(audio_path) if filename.endswith(".wav") or filename.endswith(".mp3")]

    # Ensure that number of audio files match number of captions

    min_length = min(len(audio_paths), len(column_as_list))

    audio_paths = audio_paths[:min_length]
    column_as_list = column_as_list[:min_length]

    # Verify existence of each audio path
    for path in audio_paths:
        if not os.path.exists(path):
            print(f"WARNING: The file {path} does not exist!")

    print(f"start calculating clap score, total {min_length} audios")
    # batch_size 400
    average_score = compute_average_score(model, column_as_list, audio_paths, batch_size=2)
    print(average_score)
