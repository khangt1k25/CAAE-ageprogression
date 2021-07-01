import pickle


with open('./compress/all_utk.pkl', 'rb') as f:
    all = pickle.load(f)


with open('./compress/train_utk.pkl', 'wb') as f:
    pickle.dump(all[:20000], f)


with open('./compress/test_utk.pkl', 'wb') as f:
    pickle.dump(all[20000:], f)
# dataloader = loading_dataloader('./data', image_size= 128, batch_size= 16)
