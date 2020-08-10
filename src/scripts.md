# how many disciplines there are in one year data:
grep "^Ala: " SuomiRyväsData2000|head -400|tr -s ' '|sort|uniq -c|wc -l
# how many in disciplines in the whole data:
grep "^Ala: " SuomiRyväsData200[0-3]|tr -s ' '|cut -f2,3 -d':'|sort|uniq -c|wc -l

# how many data sets in one year data (are these unanimous):
grep "^ISSN: " data/raw/SuomiRyväsData200[0-0]|wc -l
grep "^Otsikko: " data/raw/SuomiRyväsData200[0-0]|wc -l
grep "^Lehti: " data/raw/SuomiRyväsData200[0-0]|wc -l

# The number of publications in different disciplines for baseline data
CS-IS:
cat data/baseline/groundtruth-preproc_CS-AI-IS_CN.txt | grep discipline: | grep COMPUTER_SCIENCE_INFORMATION_SYSTEMS | wc -l
CS-AI:
cat data/baseline/groundtruth-preproc_CS-AI-IS_CN.txt | grep discipline: | grep COMPUTER_SCIENCE_ARTIFICIAL_INTELLIGENCE | wc -l
clinical neurology:
cat data/baseline/groundtruth-preproc_CS-AI-IS_CN.txt | grep discipline: | grep CLINICAL_NEUROLOGY | wc -l
CS-IS & CS-AI:
cat data/baseline/groundtruth-preproc_CS-AI-IS_CN.txt | grep discipline: | grep COMPUTER_SCIENCE_ARTIFICIAL_INTELLIGENCE | grep COMPUTER_SCIENCE_INFORMATION_SYSTEMS | wc -l


# Baseline data to appendix
cat data/baseline/groundtruth-preproc_CS-AI-IS_CN.txt | grep "[0-9]\{1,3\}  title:"