import torch
import torchaudio
from cpc_streaming_diarization.model import CPCStreamingDiarizationModel
from cpc_streaming_diarization.utils import get_default_device


def main():
    device = get_default_device()
    # モデルのロード
    model = CPCStreamingDiarizationModel.from_pretrained(
        "mocomoco-inc/CPCConformerTransfomerSpeakerDiarizationModel-en-2spk"
    )
    model = model.to(device)
    model.eval()

    # wav ファイルを読む
    wav, sr = torchaudio.load("example.wav")
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)  # モノラル化
    wav = wav.to(device)
    # 推論
    with torch.no_grad():
        preds = model(wav, block_size_sec=10.0, hop_size_sec=1.0)

    # 後処理
    preds = model.postprocess(preds, min_duration_on=0.3, min_duration_off=0.2)

    threshold = 0.5
    preds = (preds > threshold).int()
    print(preds)
    # output example
    # tensor([[[0, 0],
    #          [0, 0],
    #          [0, 0],
    #          ...,
    #          [0, 1],
    #          [0, 1],
    #          [0, 0]]], dtype=torch.int32)


if __name__ == "__main__":
    main()
