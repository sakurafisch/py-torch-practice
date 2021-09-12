# PyTorch Quick Start

## Install PyTorch

win
```cmd
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

linux
```zsh
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Verification
```python
import torch
x = torch.rand(5, 3)
print(x)
# tensor([[0.3380, 0.3845, 0.3217],
#         [0.8337, 0.9050, 0.2650],
#         [0.2979, 0.7141, 0.9069],
#         [0.1449, 0.1132, 0.1375],
#         [0.4675, 0.3947, 0.1426]])
torch.cuda.is_available()
# True
```

## Setup VSC Extension

[Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance)

[Prettier](https://marketplace.visualstudio.com/items?itemName=esbenp.prettier-vscode)

[IntelliCode](https://marketplace.visualstudio.com/items?itemName=VisualStudioExptTeam.vscodeintellicode)

[Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)

## Run

```zsh
python main.py
```