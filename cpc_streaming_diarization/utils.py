import torch


def get_default_device():
    """利用可能な最適なデバイスを取得"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def apply_min_duration(
    predictions: torch.Tensor,
    frame_hop_sec: float,
    min_duration_on: float,
    min_duration_off: float,
) -> torch.Tensor:
    """
    予測結果に対して後処理を適用する。
    1. min_duration_off: 短い非発話区間（ギャップ）を埋める (穴埋め)
    2. min_duration_on: 短い発話区間を除去する
    """
    if min_duration_on == 0 and min_duration_off == 0:
        return predictions

    postprocessed_preds = predictions.clone().cpu()
    num_frames, num_speakers = predictions.shape

    min_on_frames = int(min_duration_on / frame_hop_sec)
    min_off_frames = int(min_duration_off / frame_hop_sec)

    for spk_idx in range(num_speakers):
        spk_preds = postprocessed_preds[:, spk_idx]
        if min_off_frames > 0:
            changes = torch.diff(
                spk_preds, prepend=torch.tensor([0]), append=torch.tensor([0])
            )
            speech_starts = (changes == 1).nonzero(as_tuple=True)[0]
            speech_ends = (changes == -1).nonzero(as_tuple=True)[0]

            for i in range(len(speech_ends) - 1):
                gap_start = speech_ends[i]
                gap_end = speech_starts[i + 1]
                gap_duration = gap_end - gap_start
                if 0 < gap_duration <= min_off_frames:
                    postprocessed_preds[gap_start:gap_end, spk_idx] = 1
        if min_on_frames > 0:
            spk_preds_filled = postprocessed_preds[:, spk_idx]
            changes = torch.diff(
                spk_preds_filled, prepend=torch.tensor([0]), append=torch.tensor([0])
            )
            speech_starts = (changes == 1).nonzero(as_tuple=True)[0]
            speech_ends = (changes == -1).nonzero(as_tuple=True)[0]

            for i in range(len(speech_starts)):
                segment_start = speech_starts[i]
                segment_end = speech_ends[i]
                segment_duration = segment_end - segment_start
                if segment_duration < min_on_frames:
                    postprocessed_preds[segment_start:segment_end, spk_idx] = 0
    return postprocessed_preds
