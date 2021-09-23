import re
import webvtt
from gensim.summarization.summarizer import summarize as gensim_based
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import nltk
from tkinter import *
from tkinter import filedialog
import tkinter.font as tkFont
import os
import youtube_dl
import pandas as pd
import numpy as np
import spacy
nlp = spacy.load("en_core_web_sm")

def get_caption(url):
    global video_title

    ydl_opts = {
        'skip_download': True,        
        'writesubtitles': True,       
        "writeautomaticsub": True,    
        "subtitleslangs": ['en'],     
        'outtmpl': 'test.%(ext)s',    
        'nooverwrites': False,        
        'quiet': True                
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([url])
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get('title', None)
        except:
            print("Try with a YouTube URL")
    corpus = []
    for caption in webvtt.read('test.en.vtt'):
        corpus.append(caption.text)
    corpus = "".join(corpus)
    corpus = corpus.replace('\n', ' ')

    return corpus


def summarizer(text, option, fraction):
    
    frac=fraction
    if option == "Tf-Idf-Based":
        return tfidf_based(text, frac)
    if option == "Frequency-Based":
        return freq_based(text, frac)
    if option == "Gensim-Based":
        doc=nlp(text)
        text="\n".join([sent.text for sent in doc.sents])
        return gensim_based(text=text, ratio=frac)

def tfidf_based(msg,fraction):
    
    doc=nlp(msg)
    
    sents =[sent.text for sent in doc.sents]
    
    num_sent=int(np.ceil(len(sents)*fraction))
    
    
    tfidf=TfidfVectorizer(stop_words='english',token_pattern='(?ui)\\b\\w*[a-z]+\\w*\\b')
    X=tfidf.fit_transform(sents)
    
    
    df=pd.DataFrame(data=X.todense(),columns=tfidf.get_feature_names())
    indexlist=list(df.sum(axis=1).sort_values(ascending=False).index)

    needed = indexlist[:num_sent]
    
    needed.sort()
    
    summary=[]
    for i in needed:
        summary.append(sents[i])
    summary="".join(summary)
    summary = summary.replace("\n",'')
    return summary


def freq_based(text, fraction):
    
    doc = nlp(text)
    
    sentence = [sent for sent in doc.sents]
    
    numsentence = int(np.ceil(fraction*len(sentence)))

    words = [word.text.lower()
             for word in doc.doc if word.is_alpha and word.is_stop == False]
    
    df = pd.DataFrame.from_dict(
        data=dict(Counter(words)), orient="index", columns=["freq"])
    df["wfreq"] = np.round(df.freq/df.freq.max(), 3)
    df = df.drop('freq', axis=1)

    wfreq_words = df.wfreq.to_dict()

    sent_weight = []
    for sent in sentence:
        temp = 0
        for word in sent:
            if word.text.lower() in wfreq_words:
                temp += wfreq_words[word.text.lower()]
        sent_weight.append(temp)
    wdf = pd.DataFrame(data=np.round(sent_weight, 3), columns=['weight'])
    wdf = wdf.sort_values(by='weight', ascending=False)
    indexlist = list(wdf.iloc[:numsentence, :].index)

    sumlist = []
    for s in indexlist[:5]:
        sumlist.append(sentence[s])
    summary = ''.join(token.string.strip() for token in sumlist)
    return summary


root = Tk(baseName="YouTube Video Summarizer")
root.title("YouTube Video Summarizer")
root.configure(background='#d1835c')
root.geometry("600x400+400+200")
root.resizable(0, 0)

# Main Title Label
title = Label(root, text="Video Summarizer", font="bold 26",
              bg="#d1835c", padx=140, pady=10).grid(row=0, column=0)

url_label = Label(root, text="URL:", font="bold",
                  bg='#d1835c', justify="right", bd=1)
url_label.place(height=50, x=100, y=70)

model_label = Label(root, text="Model:", font="bold",
                    bg='#d1835c', justify="right", bd=1)
model_label.place(height=50, x=90, y=135)

fraction_label = Label(root, text="Fraction:", font="bold",
                       bg='#d1835c', justify="right", bd=1)
fraction_label.place(height=50, x=80, y=210)

folder_label = Label(root, text="Location:", font="bold",
                     bg='#d1835c', justify="right", bd=1)
folder_label.place(height=50, x=75, y=280)

get_url = Entry(root, width=40)
get_url.place(width=300, height=30, x=150, y=80)

options = ["TfIdf-Based", "Frequency-Based", "Gensim-Based"]

default_option = StringVar(root)
default_option.set(options[0])
drop = OptionMenu(root, default_option, *options)
drop.place(width=200, x=150, y=145)

get_fraction = Entry(root, width=40)
get_fraction.place(width=300, height=30, x=150, y=220)

get_folder = Entry(root, width=40)
get_folder.place(width=300, height=30, x=150, y=290)

folder = StringVar(root)


def browse():
    global folder
    folder = filedialog.askdirectory(initialdir='/')
    get_folder.insert(0, folder)


browse = Button(root, text="Browse", command=browse)
browse.place(height=30, x=475, y=290)

def on_clear():
    default_option.set(options[0])
    get_url.delete(0, END)
    get_folder.delete(0, END)
    get_fraction.delete(0, END)


clear = Button(root, text="Clear", command=on_clear)
clear.place(width=50, x=240, y=350)

def on_submit():
    global url, choice, frac, current, folder
    url = get_url.get()
    choice = default_option.get()
    frac = float(get_fraction.get())
    current = os.getcwd()
    folder = get_folder.get()
    os.chdir(folder)
    print(url,choice,frac,folder)
    corpus = get_caption(url)
    with open("corpus.txt",'w+') as c:
        print(corpus,file=c)
    
    summary = summarizer(corpus, choice, frac)
    filename = video_title+" "+choice+'.txt'
    filename = re.sub(r'[\/:*?<>|]', ' ', filename)
    with open(filename, 'w+') as f:
        print(summary, file=f)
    os.remove(os.getcwd()+'\\test.en.vtt')
    os.chdir(current)
    openpath = Button(root, text="Open Folder",
                      command=lambda: os.startfile(get_folder.get()))
    openpath.place(x=360, y=350)


submit = Button(root, text="Submit", command=on_submit)
submit.place(width=50, x=300, y=350)

root.mainloop()

