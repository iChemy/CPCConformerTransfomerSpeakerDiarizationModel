import torch
import torchaudio
from cpc_streaming_diarization import CPCStreamingDiarizationModel


def main():
    # モデルのロード
    model = CPCStreamingDiarizationModel.from_pretrained(
        "mocomoco-inc/speaker-diarization"
    )
    model.eval()

    # wav ファイルを読む
    wav, sr = torchaudio.load("example.wav")
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    wav = wav.mean(dim=0, keepdim=True)  # モノラル化

    # 推論
    with torch.no_grad():
        preds = model(wav)

    # 後処理
    preds = model.postprocess(preds, min_duration_on=0.3, min_duration_off=0.2)

    print(preds.shape)


if __name__ == "__main__":
    main()
