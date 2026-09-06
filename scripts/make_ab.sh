#!/bin/zsh
# usage: make_ab.sh CAND.mp3 OUT.mp3  — чередует фрагменты эталона и кандидата
C=$1; O=$2; T=$(mktemp -d)
i=0
for seg in "22 20" "100 20" "168 20"; do
  set -- $=seg; s=$1; d=$2
  ffmpeg -y -hide_banner -loglevel error -ss $s -t $d -i full.wav -af "afade=t=in:d=0.3,afade=t=out:st=$((d-0.3)):d=0.3" $T/a$i.wav
  ffmpeg -y -hide_banner -loglevel error -ss $s -t $d -i "$C" -af "afade=t=in:d=0.3,afade=t=out:st=$((d-0.3)):d=0.3" $T/b$i.wav
  ffmpeg -y -hide_banner -loglevel error -f lavfi -t 0.8 -i anullsrc=r=44100:cl=stereo $T/p$i.wav
  i=$((i+1))
done
ffmpeg -y -hide_banner -loglevel error -i $T/a0.wav -i $T/p0.wav -i $T/b0.wav -i $T/p0.wav -i $T/a1.wav -i $T/p1.wav -i $T/b1.wav -i $T/p1.wav -i $T/a2.wav -i $T/p2.wav -i $T/b2.wav \
 -filter_complex "[0][1][2][3][4][5][6][7][8][9][10]concat=n=11:v=0:a=1" -b:a 192k "$O"
rm -rf $T; echo "AB готов: $O"
