import json

# adjustable params
with open('src/utils/params.json') as json_file:
    params = json.load(json_file)

# fixed params
aux_paths = {
    'commercial':'auxiliary-data/sg-commerical-centres.csv',
    'hawker': 'auxiliary-data/sg-gov-markets-hawker-centres.csv',
    'prisch': 'auxiliary-data/sg-primary-schools.csv',
    'secsch': 'auxiliary-data/sg-secondary-schools.csv',
    'malls': 'auxiliary-data/sg-shopping-malls.csv',
    'station': 'auxiliary-data/sg-train-stations.csv',
    'demographics': 'auxiliary-data/sg-population-demographics.csv'
}
