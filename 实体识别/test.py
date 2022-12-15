import jsonlines
import json
# with jsonlines.open('./admin.jsonl')as fp:

with jsonlines.open('admin.jsonl','r')as fp:
    print(type(fp))
    for p in fp:
        print(p)
        if(p["label"]):
            print(p["text"][p["label"][0][0]:p["label"][0][1]])