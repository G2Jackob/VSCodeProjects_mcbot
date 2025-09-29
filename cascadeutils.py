import os

def gen_negative_description_file():
    
    with open('neg.txt', 'w') as f:
        for filename in os.listdir('negative'):
            f.write('negative/' + filename + '\n')

gen_negative_description_file()

#C:\Users\papil\Desktop\eeeeeeeJ\opencv\build\x64\vc15\bin\opencv_annotation.exe --annotations=pos.txt  --images=positive/

#C:\Users\papil\Desktop\eeeeeeeJ\opencv\build\x64\vc15\bin\opencv_createsamples.exe -info pos.txt -w 24 -h 24 -num 1000 -vec pos.vec


#Current Model:
#C:\Users\papil\Desktop\eeeeeeeJ\opencv\build\x64\vc15\bin\opencv_traincascade.exe -data cascade/ -vec pos.vec -bg neg.txt -w 24 -h 24 -precalcValBufSize 6000 -precalcIdxBufSize 6000 -numPos 120 -numNeg 1000 -numStages 12 -maxFalseAlarmRate 0.4 -minHitRate 0.999