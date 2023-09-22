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