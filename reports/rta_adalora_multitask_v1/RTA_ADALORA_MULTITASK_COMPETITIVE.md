# Phase B4-H Competitive Development-Only Results

Three fold-specific shared RTA + AdaLoRA adapters were trained with the frozen original fixed-padding float32 configuration. Locked-test data was not accessed.

## Direct multi-task heads

- S6: macro-F1=0.2109, balanced-accuracy=0.4230
- S3: macro-F1=0.4103, balanced-accuracy=0.5954
- S2: macro-F1=0.6047, balanced-accuracy=0.7097

## Adapted-RTA lexical fusion

- S6: macro-F1=0.2957, balanced-accuracy=0.3196
- S3: macro-F1=0.4967, balanced-accuracy=0.5454
- S2: macro-F1=0.6540, balanced-accuracy=0.6888
