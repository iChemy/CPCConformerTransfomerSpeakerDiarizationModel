# CPCConformerTransfomerSpeakerDiarizationModel

This repository provides the **PyTorch implementation** of a streaming-capable **speaker diarization model** based on [BW-EDA-EEND](https://doi.org/10.48550/arXiv.2011.02678).  
The model is trained for **two-speaker English audio**, using a Conformer encoder and CPC (Contrastive Predictive Coding) feature extractor.

👉 Pretrained models and usage examples are available on Hugging Face:  
[mocomoco-inc/SpeakerDiarizationModel-en-2spk](https://huggingface.co/mocomoco-inc/SpeakerDiarizationModel-en-2spk)

---

## Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/mocomoco-inc/CPCConformerTransfomerSpeakerDiarizationModel.git
cd CPCConformerTransfomerSpeakerDiarizationModel
pip install -e .
```

---

## Repository Structure
```
.
├── cpc_streaming_diarization
│   ├── config.py   # Model configuration classes and default parameters
│   ├── model.py    # Main diarization model (CPC + Conformer + Transformer)
│   ├── modules     # Submodules used inside the model
│   │   └── ...
│   ├── utils.py    # Helper functions (e.g., device setup, postprocessing)
│   └── ...
├─ examples
│   └── diarize.py  # Example script for running inference
└── ...
```

---

## Pretrained Models
- Hugging Face Hub: [mocomoco-inc/SpeakerDiarizationModel-en-2spk](https://huggingface.co/mocomoco-inc/SpeakerDiarizationModel-en-2spk)

---

## License
This project is licensed under the [Apache-2.0 License](./LICENSE).

---
## Contact
For any inquiries, please contact us at: <br />
mocomoco inc. Inada Bldg. 302, 7-20-19 Roppongi,<br />
Minato-ku, Tokyo 106-0032, Japan<br />
contact@mocomoco.ai