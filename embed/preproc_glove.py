import io
import pickle
import numpy as np
glove_filename = '../../coco_data2014/glove.840B.300d.txt'
with io.open(glove_filename,'r',encoding='utf-8') as f:
    lines = f.readlines()

# code from HW5 to open glove dictionary
###########################
glove_dictionary = []
glove_embeddings = []
count = 0
for line in lines:
    line = line.strip()
    line = line.split(' ')
    glove_dictionary.append(line[0])
    embedding = np.asarray(line[1:],dtype=np.float)
    glove_embeddings.append(embedding)
    count+=1
    if(count>=100000):
        break

glove_dictionary = np.asarray(glove_dictionary)
glove_embeddings = np.asarray(glove_embeddings)
############################



# add columns for start, stop, unknown, and mask 
add_col = np.random.uniform(0, 0.5, (4, 300))
glove_embeddings = np.concatenate((add_col,glove_embeddings))
word_to_id = {token: idx+4 for idx, token in enumerate(glove_dictionary)}

# add start/end/unknown token
word_to_id['<S>'] = 1
word_to_id['</S>'] = 2
word_to_id['UNK'] = 3

# open our self-created dictionary
with open('../word_to_id.p', 'rb') as fp:
    orig_dictionary = pickle.load(fp)
orig_id_to_word = np.load('../id_to_word.npy')
counter = 0
counter2 = 0
# sort word_to_id to match our self-created (non-glove) dictionary
new_embed = np.zeros((orig_id_to_word.shape[0], glove_embeddings.shape[1])) 
for token in orig_dictionary:
    if word_to_id.get(token) is not None:
        new_embed[orig_dictionary.get(token), :] = glove_embeddings[word_to_id.get(token), :]
    else:
        counter += 1
        new_embed[orig_dictionary.get(token), :] = np.random.uniform(0, 0.5, 300)
        if orig_dictionary.get(token)<=9221:
            counter2 += 1
new_embed[0, :] = glove_embeddings[0, :]  # for the mask

np.save('glove_embeddings.npy', new_embed)
print('number of words missing from glove embedding.... ', counter)
print('number of words missing from glove embedding (<=9221).... ', counter2)



