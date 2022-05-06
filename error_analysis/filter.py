# -*- coding: utf-8 -*-

###################################### IMPORTANT INFORMATION ######################################
"""
This program filters the data produced by error_measure.py according to the parameters defined below.
It creates a folder called 'filtered_runs' in the same path where it saves the data. It will erase all
previous filtered data files before creating new ones.

The data is saved under the error_analysis/error_data/ folder.

Parameters to adjust manually:
    GENERAL PARAMETERS
    network_arch --> '1sc', '2sc', '1sd', '2sd'. Network architecture
    learning_rate --> float, Learning rate of the runs we want to filter.
    E_min --> float, (mean) Energy below which we stop accepting runs.
    good_runs_min --> int, Minimum number of good (filtered) runs we wish to end up with. If this is
                      not the case, a message is printed via console with additional information.
    hidden_neurons_list --> list, Number of neurons list that we wish to keep. (Note that we may have
                            trained models with a different number of nhid)
    seed --> int, Random seed.
"""

############################## ADJUSTABLE PARAMETERS #################################
# General parameters
network_arch = '1sc'
learning_rate = 0.01 # Use decimal notation.
E_min = -2.22
good_runs_min = 20
hidden_neurons_list = [20,30,40,60,80,100]
seed = 1

###################################### IMPORTS ######################################
import pathlib,os,sys,re,shutil,random
initial_dir = pathlib.Path(__file__).parent.resolve()
os.chdir(initial_dir)
sys.path.append('..')

############################ FILES, DIRECTORIES ############################
random.seed(seed)
net_arch_name_map = {'1sc':'fully_connected_ann','2sc':'fully_connected_ann',
                     '1sd':'separated_ann','2sd':'separated_ann'}
path_to_error_files = f'error_data/nlayers{network_arch[0]}/{net_arch_name_map[network_arch]}/Sigmoid/lr{learning_rate}/'
os.chdir(path_to_error_files)

"""
We prepare a directory called 'filtered_runs' to which we copy all error files with
'nhid' in 'hidden_neurons_list'.
"""
 
all_files = os.listdir()
list_of_error_files = []
files_not_wanted = []

# We select the error files among all the files in this directory
for file in all_files:
    if len(file.split('.'))>1:
        if file.split('.')[1] == 'txt' and file[:4] == 'nhid':
            nhid = int(file.split('.')[0].replace('nhid',''))
            if nhid in hidden_neurons_list:
                list_of_error_files.append(file)
            else:
                files_not_wanted.append(file)
                print('Skipping error file with nhid = {} ...'.format(nhid))
            
# We sort the list of error files by nhid
def num_sort(test_string):
    return list(map(int, re.findall(r'\d+', test_string)))[0]
list_of_error_files.sort(key=num_sort)            

# If the filtered_runs folder does not exist, we create it 
if not os.path.exists('filtered_runs'):
    os.makedirs('filtered_runs')
    print('Creating directory "filtered_runs"...')
    
# We make a copy of the error files into the filtered_runs folder
for file in list_of_error_files:
    shutil.copy(file,'filtered_runs/')

os.chdir('filtered_runs')

# We remove those files with nhid not in hidden_neurons_list
for file in os.listdir():
    if file in files_not_wanted:
        os.remove(file)
        print('File {} removed from the filtered_runs folder.'.format(file))


############################ FILTERING, COUNTING ############################
"""
We create new error files that contain only good_runs_min good runs tops, with all 
single energies being < E_min. 
"""

# We go through each of the files and we keep only the lines with E<-2.22 MeV
deleted_Es = []
incomplete_files = 0
for file in os.listdir():
    if file.split('.')[1] == 'txt' and file[:4] == 'nhid':
        good_runs = 0
        
        # We store the data of the file in a list
        with open(file, "r") as f:
            lines = f.readlines()
            
        # We destroy the file, writing only the lines that fulfil our criteria
        with open(file, "w") as f:
            for line in lines:
                E = line.strip('\n').split(' ')[0]
                if float(E) <= E_min:
                    f.write(line)
                    good_runs += 1
                else:
                    deleted_Es.append(E)
            # We print a warning whenever the file has insufficient data
            if good_runs < good_runs_min:
                print('The file {}filtered_runs/{} only has {} good runs.'.format(path_to_error_files,file,good_runs))
                incomplete_files += 1
        
        # We shuffle the lines of the files with good_runs > good_runs_min
        if good_runs > good_runs_min:
            with open(file, "r") as f:
                lines = f.readlines()
                random.shuffle(lines)
                shuffled_data = []
                for line in lines:
                    if len(shuffled_data) < good_runs_min: 
                        shuffled_data.append(line)
                    else:
                        break
                    
            # We open the same file and write only the first good_runs_min lines, 
            # which is a random subset of the initial good_runs
            with open(file, 'w') as f:
                for line in shuffled_data:
                    f.write(line)
        
if incomplete_files == 0:
    print('All files had already at least {} good runs!'.format(good_runs_min))