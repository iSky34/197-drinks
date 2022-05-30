'''
Runs an inference on a single audio file.
Assumption is data file and checkpoint are in the same args.path
Simple test:
    python3 kws-infer.py --wav-file <path-to-wav-file>  
To use microphone input with GUI interface, run:
    python3 kws-infer.py --gui
    On RPi 4:
    python3 kws-infer.py --rpi --gui
Dependencies:
    sudo apt-get install libasound2-dev libportaudio2 
    pip3 install pysimplegui
    pip3 install sounddevice 
    pip3 install librosa
    pip3 install validators
Inference time:
    0.03 sec Quad Core Intel i7 2.3GHz
    0.08 sec on RPi 4
'''


import torch
import argparse
import torchaudio
import os
import numpy as np
import librosa
import sounddevice as sd
import time
import validators
from torchvision.transforms import ToTensor
from einops import rearrange

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="data/speech_commands/")
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--n-mels", type=int, default=64)
    parser.add_argument("--win-length", type=int, default=None)
    parser.add_argument("--hop-length", type=int, default=252)
    parser.add_argument("--wav-file", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default="https://github.com/roatienza/Deep-Learning-Experiments/releases/download/models/resnet18-kws-best-acc.pt")
    parser.add_argument("--gui", default=False, action="store_true")
    parser.add_argument("--rpi", default=False, action="store_true")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    return args


# main routine
if __name__ == "__main__":
    CLASSES = ['silence', 'unknown', 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow',
               'forward', 'four', 'go', 'happy', 'house', 'learn', 'left', 'marvin', 'nine', 'no',
               'off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three',
               'tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero']

    idx_to_class = {i: c for i, c in enumerate(CLASSES)}

    args = get_args()

    # if validators.url(args.checkpoint):
    #     checkpoint = args.checkpoint.rsplit('/', 1)[-1]
    #     # check if checkpoint file exists
    #     if not os.path.isfile(checkpoint):
    #         torch.hub.download_url_to_file(args.checkpoint, checkpoint)
    # else:
    #     checkpoint = args.checkpoint
    checkpoint="kwstransform.pt"

    print("Loading model checkpoint: ", checkpoint)
    scripted_module = torch.jit.load(checkpoint)

    if args.gui:
        import PySimpleGUI as sg
        sample_rate = 16000
        sd.default.samplerate = sample_rate
        sd.default.channels = 1
        sg.theme('DarkAmber')
  
    elif args.wav_file is None:
        # list wav files given a folder
        print("Searching for random kws wav file...")
        label = CLASSES[2:]
        label = np.random.choice(label)
        path = os.path.join(args.path, "SpeechCommands/speech_commands_v0.02/")
        path = os.path.join(path, label)
        wav_files = [os.path.join(path, f)
                     for f in os.listdir(path) if f.endswith('.wav')]
        # select random wav file
        wav_file = np.random.choice(wav_files)
    else:
        wav_file = args.wav_file
        label = args.wav_file.split("/")[-1].split(".")[0]

    if not args.gui:
        waveform, sample_rate = torchaudio.load(wav_file)

    transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                     n_fft=512,
                                                     win_length=args.win_length,
                                                     hop_length=252,
                                                     n_mels=64,
                                                     power=2.0)
    if not args.gui:
        mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        mel = mel.unsqueeze(0)

        pred = torch.argmax(scripted_module(mel), dim=1)
        print(f"Ground Truth: {label}, Prediction: {idx_to_class[pred.item()]}")
        exit(0)

    layout = [ 
        [sg.Text('Say it!', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 140), key='-OUTPUT-'),],
        [sg.Text('', justification='center', expand_y=True, expand_x=True, font=("Helvetica", 100), key='-STATUS-'),],
        [sg.Text("Options are 'backward', 'bed', 'bird', 'cat', 'dog', 'down', 'eight', 'five', 'follow','forward', 'four', 'go', 'happy','yes','zero'",justification='left', expand_y=True, expand_x=True, font=("Helvetica", 20))],
        [sg.Text ("'house', 'learn', 'left', 'marvin', 'nine', 'no','off', 'on', 'one', 'right', 'seven', 'sheila', 'six', 'stop', 'three','tree', 'two', 'up', 'visual', 'wow', 'yes', 'zero'", justification='left', expand_y=True, expand_x=True, font=("Helvetica", 20))],
        
        [sg.Text('Speed', expand_x=True, font=("Helvetica", 28), key='-TIME-')],
        [sg.Text('Total Time:', expand_x=True,justification='right', font=("Helvetica", 28), key='-TTIME-')],
    ]

    window = sg.Window('KWS Inference', layout, location=(0,0), resizable=True).Finalize()
    window.Maximize()
    window.BringToFront()

    total_runtime = 0
    n_loops = 0
    a=time.time()
    while True:
        
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        event, values = window.read(100)
        if event == sg.WIN_CLOSED:
            break
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        waveform = sd.rec(sample_rate).squeeze()
        
        sd.wait()
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,
                                                n_fft=512,
                                                win_length=None,
                                                hop_length=252,
                                                n_mels=64,
                                                power=2.0)
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        if waveform.max() > 1.0:
            continue
        start_time = time.time()
        if args.rpi:
            # this is a workaround for RPi 4
            # torch 1.11 requires a numpy >= 1.22.3 but librosa 0.9.1 requires == 1.21.5
            waveform = torch.FloatTensor(waveform.tolist())
            mel = np.array(transform(waveform).squeeze().tolist())
            mel = librosa.power_to_db(mel, ref=np.max).tolist()
            
            mel = torch.FloatTensor(mel)
            mel = mel.unsqueeze(0)

        else:
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            mel = ToTensor()(librosa.power_to_db(transform(waveform).squeeze().numpy(), ref=np.max))
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        

        #   def __init__(self,path, batch_size, num_workers=32, patch_num=4, n_fft=512,n_mels=64, win_length=None, hop_length=252, class_dict={}, **kwargs):
   
        ##mel = ToTensor()(librosa.power_to_db(
        #  transform(waveform).squeeze().numpy(), ref=np.max))
        #mels = rearrange(mels, 'b c (p1 h) (p2 w) -> b (p1 p2) (c h w)', p1=self.patch_num, p2=self.patch_num)
        if waveform.shape[-1] < sample_rate:
            waveform = torch.cat([waveform, torch.zeros((1, sample_rate - waveform.shape[-1]))], dim=-1)
        
        elif waveform.shape[-1] > sample_rate:
            waveform = waveform[:,:sample_rate]
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
                    # mel from power to db
      #  mel.append(ToTensor()(librosa.power_to_db(transform(waveform).squeeze(0).numpy(), ref=np.max)))
            #mels.append(self.transform(waveform))
        #     labels.append(torch.tensor(self.class_dict[label]))
        #     labels.append(torch.tensor(self.class_dict[label]))
        #    wavs.append(waveform)

      #  mel = torch.stack(mel, dim=0)
        # labels=torch.stack(labels)
        #  labels = torch.LongTensor(labels)


        mel = rearrange(mel, 'c ( h) (p2 w) ->  (p2) (c h w)', p2=16)
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        
            
            
            
            
            
            
            
            
            
            
            
           
        mel = mel.unsqueeze(0)
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")
        pred = scripted_module(mel)
        pred = torch.functional.F.softmax(pred, dim=1)
        max_prob =  pred.max()
        elapsed_time = time.time() - start_time
        total_runtime += elapsed_time
        n_loops += 1
        ave_pred_time = total_runtime / n_loops
        if max_prob > args.threshold:
            pred = torch.argmax(pred, dim=1)
            human_label = f"{idx_to_class[pred.item()]}"
            window['-OUTPUT-'].update(human_label)
            window['-OUTPUT-'].update(human_label)
            if human_label == "stop":
                window['-STATUS-'].update("Goodbye!")
                # refresh window
                window.refresh()
                time.sleep(1)
                break
               
        else:
            window['-OUTPUT-'].update("...")
        
        window['-TIME-'].update(f"{ave_pred_time:.2f} sec")
        window['-TTIME-'].update(f"total time: {round(time.time()-a,2)} sec")


    window.close()
