# Malicious JavaScript Detection

## Running the demo

```bash
docker-compose up -d
```

Example request body:

```json
{
	"javascript": [
		{
			"idx": "string1",
			"code": "console.log('Hello World!');"
		},
		{
			"idx": "string2",
			"code": "document.write('<center>' '<iframe width=\"11\" height=\"1\" ' 'src=\"http://laghzesh.rzb.ir\" ' 'style=\"border: 0px;\" ' 'frameborder=\"0\" ' 'scrolling=\"auto\">' '</iframe>');"
		}
	]
}
```

Response:

```json
{
  "results": [
    {
      "idx": "string1",
      "label": "benign"
    },
    {
      "idx": "string2",
      "label": "malicious"
    }
  ]
}
```

## Training

```bash
./scripts/train_codebert_unimodal.sh
./scripts/train_codebert_bimodal.sh
```

## Testing

```bash
./scripts/inference_codebert_unimodal.sh
./scripts/inference_codebert_bimodal.sh
```

## Setting up remote machine on Vast.ai

1. Use the `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel` image
2. SSH into the remote server, then install poetry and enable conda:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/install.python-poetry.org/main/install-poetry.py | python3
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc

# enable conda env
conda init

# restart terminal by sourcing bashrc
source ~/.bashrc

# export requirements.txt
poetry export -f requirements.txt --without-hashes --output requirements.txt

pip install -r requirements.txt

# config git
git config --global user.email "truong173@outlook.com"
git config --global user.name "truonghm"

# clone this repo
git clone https://github.com/truonghm/malicious-code-detection.git
cd malicious-code-detection
```

Then open the repo in a VSCode remote window.