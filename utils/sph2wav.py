import os
from sphfile import SPHFile
from tqdm import tqdm

path = '../data/switchboard/audio/'  # Path of folder containing .sph files
output_path = '../data/switchboard/audio_wav/'
folder = os.fsencode(path)

filenames = []
folderpath = []
outputfile = []

for file in os.listdir(folder):
    filename = os.fsdecode(file)
    if filename.endswith( ('.sph') ): # whatever file types you're using...
        filenames.append(filename)

length = len(filenames) 

print("Total files: ", length)

for i in range(length):
	fpath = os.path.join(path+filenames[i])
	folderpath.append(fpath)
	outpath = os.path.join(output_path+filenames[i].split('.')[0]+".wav")	
	outputfile.append(outpath)


for i in tqdm(range(length), desc="Converting to WAV..."):
	sph =SPHFile(folderpath[i])
	print(sph.format)
	sph.write_wav(outputfile[i]) # Customize the period of time to crop



	
	
