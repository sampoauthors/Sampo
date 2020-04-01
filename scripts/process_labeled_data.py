import sys
import pandas as pd

dat = pd.read_csv(sys.argv[1])
dat = dat[~dat['_golden']]

# find junks
all_junks = set()
for _, row in dat.iterrows():
    label = row['whats_the_relationship_between_the_two_phrases']
    if label == 'aisjunk':
        all_junks.add((row['modifier'], row['aspect']))
    elif label == 'bisjunk':
        all_junks.add((row['nn_modifier'], row['nn_aspect']))

pre_mod, pre_asp, con_mod, con_asp, pos, rpos = [], [], [], [], [], []
junk_row_list = []

for _, row in dat.iterrows():
    label = row['whats_the_relationship_between_the_two_phrases']
    junk_row = False
    if (row['modifier'], row['aspect']) in all_junks:
        junk_row = True
    elif (row['nn_modifier'], row['nn_aspect']) in all_junks:
        junk_row = True
    junk_row_list.append(junk_row)
    #junk_row_list.append(junk_row)
    #if label in ['aisjunk', 'bisjunk']:
    #    continue
    pre_mod.append(row['modifier'])
    pre_asp.append(row['aspect'])
    con_mod.append(row['nn_modifier'])
    con_asp.append(row['nn_aspect'])
    #con_mod.append(row['modifier'])
    #con_asp.append(row['aspect'])
    #pre_mod.append(row['nn_modifier'])
    #pre_asp.append(row['nn_aspect'])
    if label == 'atob':
        pos.append(True)
        #pos.append(False)
        rpos.append(True)
        #rpos.append(True)
    elif label == 'btoa':
        pos.append(False)
        #pos.append(True)
        rpos.append(True)
        #rpos.append(True)
    elif label == 'same':
        pos.append(True)
        #pos.append(True)
        rpos.append(True)
        #rpos.append(True)
    else:
        pos.append(False)
        #pos.append(False)
        rpos.append(False)
        #rpos.append(False)

new_dat = pd.DataFrame({'modifier': pre_mod, 'aspect': pre_asp, 'other_modifier': con_mod, 'other_aspect': con_asp, 'label': pos, 'rel_label': rpos, 'junk_row': junk_row_list})
new_dat.to_csv(sys.argv[1][:-4] + '_cleaned.csv', index=False)
