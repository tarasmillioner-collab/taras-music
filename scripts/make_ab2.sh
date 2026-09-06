#!/bin/zsh
C=$1; O=$2; T=$(mktemp -d)
i=0
for seg in "0 18" "100 18"; do
  set -- $=seg; s=$1; d=$2
  ffmpeg -y -hide_banner -loglevel error -ss $s -t $d -i full.wav -af "afade=t=out:st=$((d-0.3)):d=0.3" $T/a$i.wav
  ffmpeg -y -hide_banner -loglevel error -ss $s -t $d -i "$C" -af "afade=t=out:st=$((d-0.3)):d=0.3" $T/b$i.wav
  i=$((i+1))
done
ffmpeg -y -hide_banner -loglevel error -f lavfi -t 0.8 -i anullsrc=r=44100:cl=stereo $T/p.wav
ffmpeg -y -hide_banner -loglevel error -i $T/a0.wav -i $T/p.wav -i $T/b0.wav -i $T/p.wav -i $T/a1.wav -i $T/p.wav -i $T/b1.wav \
 -filter_complex "[0][1][2][3][4][5][6]concat=n=7:v=0:a=1" -b:a 192k "$O"
rm -rf $T; echo "AB: $O"
