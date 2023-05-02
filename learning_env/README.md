## Alias 설정

vi ~/.bashrc

alias dn='docker run --gpus all -itd -p 8000:8000 -v /mnt/f/AIProject:/home/jupyter --name deep_notebook sjseo85/deep_notebook'

source ~/.bashrc
