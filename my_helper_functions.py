# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 20:18:15 2017

@author: Tichakunda Mangono
"""

# In the future, consider the following..
# Class for datetimes already exists
# Class for places (locations, countries, continent etc. use googlemaps!)
# Plotting Maps etc.
# Class for preprocessing data e.g. data cleaning and EDA
# All my useful functions in one place!

# First, import the coutry code to continent map for my location functions
import pandas as pd
import os
path = os.curdir+"\Data\Source\\"
ct_df = pd.read_csv(path+"country_code_to_continent_map.csv")
CONTINENT_DICT = {x:y for x,y in zip(ct_df.country,ct_df.continent)}

###############################################################################
# DATA CLEANING & OTHER BASIC FUNCTIONS #
# Note: Some functions are for PEPFAR Data ONLY#
###############################################################################

def getAddress(entity, continent_dict=CONTINENT_DICT):
    """
    Takes in the name of a place and returns a tuple of the formated address, 
    country (short, and long versions) and continent
    """
    import urllib
    import requests
    
    # Find the URL to communicate with google
    main_api = "http://maps.googleapis.com/maps/api/geocode/json?"
    address = entity #'Charles River Associates, Washington, DC'
    
    # urlencode puts the %20 for the spaces, makes the data validation nicer
    url = main_api + urllib.parse.urlencode({'address': address})
   
    # Now to go and get jsson response
    json_data = requests.get(url).json()
    #json_status = json_data['status']
      
    # Get address
    formatted_address = json_data['results'][0]['formatted_address']
    # Get continent
    items_bool = [x['types'] == ['country', 'political'] for x in json_data['results'][0]['address_components']]
    i = items_bool.index(True)
    
    country_s = json_data['results'][0]['address_components'][i]['short_name']
    country_l = json_data['results'][0]['address_components'][i]['long_name']
    continent = continent_dict[country_s]
    return formatted_address, country_l, continent
    #getAddress('Harvard, MA')
        
def plotFreq(data, column_list, cutoff = -1):
    """
    Takes dataframe, list of target columns and a cutoff 
    e.g. plot the top 20 frequencies if cut0ff=20
    defaults to cutoff = -1 to plot the whole distribution.
    Returns charts showing distribution for each of the columns. 
    Frequency of observations on the y-axis.
    """
    import matplotlib.pyplot as plt
    l = len(column_list)/2+1
    m = 2
    plt.figure(figsize=(12,l*4))
    for i in range(len(column_list)):
        plt.subplot(l,m,i+1)
        unique = len(data[column_list[i]].value_counts())
        tit = column_list[i].upper() + " Freq of " + str(unique) + " Unique"
        if cutoff == -1:
            data[column_list[i]].value_counts().plot()
        else:
            data[column_list[i]].value_counts()[:cutoff].plot(kind='barh')
        plt.title(tit,x=0.6,y=0.6)
        
def getTransitMetrics(origin, destination, gmaps_api_key):
    """
    Takes origin and destination and returns distance in kilometers, and time in hours
    """
    import googlemaps
                
    # Find the URL to communicate with google
    #link here: https://developers.google.com/maps/documentation/distance-matrix/
    #main_api = 'https://maps.googleapis.com/maps/api/distancematrix/json?'
    
    gmaps = googlemaps.Client(key=gmaps_api_key)
    results = gmaps.distance_matrix(origin, destination)
    dist = results['rows'][0]['elements'][0]['distance']['value']/float(1000)
    time = results['rows'][0]['elements'][0]['duration']['value']/float(3600)
    print(dist, time)
    return dist, time

#getTransitMetrics('Chikato Primary School', 'Masvingo General Hospital')

def pdfMerge(filenames, source_directory, merged_filename):
    """
    Purpose: Uses PyPDF library to merge files.
    Inputs: filenames - List of strings. pdf filenames of the source pdfs to be merged in the same directory
            source_directory - string, full pathname of the directory housing these files
            merged_filename - target filename of the merged pdf document
    Returns: A merged pdf document saved in the same directory as source files
    """
    import os
    os.chdir(source_directory)
    
    from PyPDF2 import PdfFileMerger, PdfFileReader
    merger = PdfFileMerger(strict=False)
    for filename in filenames:
        merger.append(PdfFileReader(open(filename, 'rb')))
    merger.write(merged_filename)

def getColumnDataTypes(data):
    """
    Takes dataframe as input, returns columns and datatypes summary. 
    Category vs. Numerical columns or Discrete vs. Continuous. Also Time Series/Date
    
    # Notes..df.select_dtypes(include=[], exclude=[])
    To select all numeric types use the numpy dtype numpy.number
    To select strings you must use the object dtype, but note that this will return all object dtype columns
    See the numpy dtype hierarchy
    To select datetimes, use np.datetime64, ‘datetime’ or ‘datetime64’
    To select timedeltas, use np.timedelta64, ‘timedelta’ or ‘timedelta64’
    To select Pandas categorical dtypes, use ‘category’
    To select Pandas datetimetz dtypes, use ‘datetimetz’ (new in 0.20.0), or a ‘datetime64[ns, tz]’ string
    """
    # Isolate the different types of colums
    # Numeric/Continuous
    num_cols = list(data._get_numeric_data().columns)

    # Categorical/discrete
    cat_d = data.select_dtypes(include=['object', 'category'])    
    cat_cols = list(cat_d.columns)
    # Date-time
    dat_d = data.select_dtypes(include=['datetime', 'datetime64','timedelta', 'timedelta64'])
    date_cols = list(dat_d.columns)
    
    t, n, c, d = len(data.columns), len(num_cols), len(cat_cols), len(date_cols)
    print(t,n,c,d)
    # Check that we have all 33 columns covered
    if n+c+d == t:
        print("ALL columns accounted for!")
        print ("Total columns: {} \n, Numeric columns: {}\n, Categorical columns: {}\n, Datetime columns: {}".format(t, n, c, d))
    else:
        print("Some columns NOT accounted for...")
        print ("Total columns: {} \n, Numeric columns: {}\n, Categorical columns: {}\n, Datetime columns: {}".format(t, n, c, d))
    return (num_cols, cat_cols, date_cols)

def parse_raw_data(ExcelFileObject):
    # Some notes:  a problem with the encoding, solution here
    #..https://stackoverflow.com/questions/19699367/unicodedecodeerror-utf-8-codec-cant-decode-byte
    # raw = pd.read_csv('../Supply Chain/pepfar supply chain.csv', encoding = "ISO-8859-1")
    # Summary data
    full = ExcelFileObject
    summary = full.parse(sheetname = full.sheet_names[0])
    summary.name='summary'
    # Purpose data
    purpose = full.parse(sheetname = full.sheet_names[1])
    purpose.name = 'purpose'
    # Ref is the data dictionary for reference on what the columns mean
    ref = full.parse(sheetname = full.sheet_names[2]).iloc[:33,:]
    ref.name = 'ref'
    # Data if the full data set
    data = full.parse(sheetname = full.sheet_names[3])
    data.name = 'data'
    parsed_data = [summary, purpose, ref, data]
    return parsed_data

def rename_data_columns(data, newcol_list):
    # See the original columns
    print("Old columns: ",data.columns)
    newcol_dict = dict(zip(data.columns, newcol_list))
    data.rename(columns=(newcol_dict), inplace =True)
    print("New columns: ", data.columns)
    return data

def getReferenceInfo(data, column, ref):
    fn = ref[ref['NewColumn'] == column]['FieldName']
    dt = ref[ref['NewColumn'] == column]['DataType']
    fd = ref[ref['NewColumn'] == column]['FieldDescription']
    ft = ref[ref['NewColumn'] == column]['FieldNotes']
    print("{} \n=======\n, {} \n=======\n, {} \n=======\n, {} \n=======\n\
          Examples: \n{} \n=======\n".format(fn, dt, fd, ft, data[column].head())) 
    
# Separate into blocks by datatype. See the types
def get_blocks_by_dtype(data):
    " Gives column type report, returns separated blocks by data type"
    blocks = data.as_blocks()
    print("Total Number of Columns: {}\nBreakdown....\n".format(len(data.columns)))
    for k in blocks.keys():
        print("Type: {} , Count: {} \nColumns and null counts---: \n{}\n".format(
            k,len(blocks[k].columns),blocks[k].isnull().sum()))
    return blocks
    
def clean_data():
    # To implement, this should run another file which does data cleaning
    pass

def load_clean_data(names_):
    """
    Takes a list of names as parameters , returns a dictionary of dataframes for each of the data pieces available
    """
    chunky_keys = names_
    dnames = ["_"+str(i)+"_"+chunky_keys[i]+".csv" for i in range(len(chunky_keys))]  
#               ['_0_dnum.csv','_1_dnum_country.csv', '_2_dnum_vendor.csv', '_3_dnum_factory.csv'
#               , '_4_dnum_brand.csv', '_5_dnum_molecule_test.csv','_6_dnum_lpifsi.csv',
#               '_7_ddate.csv' , '_8_dobject.csv' ]
    #encoding = "ISO-8859-1"
    # Load in all of the feature datasets to date
    #dchunks = [None,None,None,None,None,None,None,None,None]
    chunky_dict = {x: None for x in chunky_keys}
    path = os.curdir+'\Data\Features\\'
    for i in range(len(dnames)):
        try:
            print("trying normal method for: ... ", i)
            chunky_dict[chunky_keys[i]] = pd.read_csv(path+dnames[i])
        except UnicodeDecodeError:
            print("Failed with encoding error, trying again for: ... ", i)
            chunky_dict[chunky_keys[i]] = pd.read_csv(path+dnames[i], encoding = "ISO-8859-1")
        print("Sucess for: ... ", i)
    # Drop the extra column
    for d in chunky_dict.values():
        d.drop('Unnamed: 0', axis=1, inplace=True)
    return chunky_dict


######################################################################################
# FEATURE ENGINEERING #
# Feature Creation for Country Stability, Logistics Index, and Factory Location #
######################################################################################

#import my_helper_functions as mhf
import pandas as pd
#import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Load in the main data to be used for merging at the end..
# main_data = mhf.load_clean_data()

#### 1. Country stability Index   ######

def generate_country_stability_features():
    """
    Cleans and transforms data from fragile state index in excel sheets. Returns cleaned
    data frame
    """
    # From fragile state data (FundForPeace)
    # Read in the files 
    import os
    path = os.curdir+'\Data\Source\\'
    fsi_n = ['fsi-2006.xlsx','fsi-2007.xlsx','fsi-2008.xlsx','fsi-2009.xlsx','fsi-2010.xlsx','fsi-2011.xlsx'
    ,'fsi-2012.xlsx','fsi-2013.xlsx','fsi-2014.xlsx','fsi-2015.xlsx','fsi-2016.xlsx','fsi-2017.xlsx']
    fsi_xl = [pd.read_excel(pd.ExcelFile(path+f)) for f in fsi_n]

    # trim the dataframe
    fsi_dfs = [d[['Country','Year','Total']] for d in fsi_xl]
    for df in fsi_dfs:
        df['year'] = df['Year'].apply(lambda x: str(x.year))
        df['country'] = df['Country'].str.strip()
        df.rename(columns={'Total':'fsi'}, inplace=True)
        df.drop(['Year', 'Country'], axis=1, inplace=True)

    # Now concatenate vertically!
    [len(d) for d in fsi_dfs]

    # concatenate to final 
    fsi_all = pd.concat(fsi_dfs, axis=0)

    # Save to disk , ready for use. Consider joining all country statistics at some point?
    # Both origin and destination
    #fsi_all.to_csv('fsi_2006-2017.csv')
    return fsi_all

###### 2. Logistics Index   ######

# series to be combined into data frames later
def compare_columns(data1, data2,column1,column2):
    """
    Takes two dataframes and a column from each. Returns dataframe of the two columns, and prints out the 
    comparisons.
    """
    fsi_all,lpi_all = data1, data2
    fsi_c, lpi_c = pd.Series(fsi_all[column1].unique(), name='fsi'),pd.Series(lpi_all[column2].unique(), name='lpi')
    print("data1 shape:",len(fsi_c),"data2 shape:",len(lpi_c))
    # merge countries
    df_country = pd.merge(pd.DataFrame(fsi_c),pd.DataFrame(lpi_c), left_on='fsi', right_on='lpi'
             , suffixes=('_fsi', '_lsi'), how='outer')
    print(df_country.shape)
    #list
    lpi_not_fsi = list(df_country[df_country.fsi.isnull()].lpi)
    fsi_not_lpi = list(df_country[df_country.lpi.isnull()].fsi)
    print(len(lpi_not_fsi), len(fsi_not_lpi))
    print("In data2 but not in data1: {} \n----\nIn data1 but not in data2: {}".format(
        lpi_not_fsi, fsi_not_lpi))
    return df_country

def generate_country_logistics_features():
    """
    Cleans and transforms excel file with several logistics indicators for countries
    Returns cleaned dataframe 
    """
    # load the data
    import os
    path = os.curdir+'\Data\Source\\'
    lpi = pd.ExcelFile(path+'International_LPI_from_2007_to_2016.xlsx')

    # iterate over dfs, pull and rename
    lpi_dfs = []
    for s in lpi.sheet_names:
        df_ =lpi.parse(sheetname=s).reset_index()
        df_.columns = [s+"&"+c for c in df_.columns]
        df_.rename(columns={s+'&'+'country':'country',s+'&'+'index':'index'}, inplace=True)
        lpi_dfs.append(df_.drop('index', axis=1))
    [d.shape for d in lpi_dfs]

    lpicountries=[]
    for d in lpi_dfs:
        lpicountries += list(lpi_dfs[0].country)
    lpicountries = set(lpicountries)
    print(len(lpicountries))
    # Rename and prepare to merge all dfs horizontally
    ct = pd.DataFrame([], lpicountries).reset_index().rename(columns={'index':'country'})
    # Iterate and merge
    for d in lpi_dfs:
        ct = ct.merge(d,how='left',left_on='country', right_on='country' )
    print("shape ct: ",ct.shape)

    lpi_cols = list(set([x.split('&')[1] for x in ct.columns[1:]]))
    mis = ['2006', '2008', '2009', '2011', '2013', '2015']
    missing_cols = [x+'&'+y for x in mis for y in lpi_cols]
    print("missing_cols:", len(missing_cols))
    ''' Mapping...
    2006 --> 2007
    2008 --> (2010-2007)/3+2007
    2009 --> 2*(2010-2007)/3+2007
    2011 --> (2010+2012)/2
    2013 --> (2012+2014)/2
    2015 --> (2014+2016)/2
    '''
    y='2006'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = ct['2007&'+attr]
    y='2008'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = (ct['2010&'+attr]-ct['2007&'+attr])/3.0 + ct['2007&'+attr]
    y='2009'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = 2*(ct['2010&'+attr]-ct['2007&'+attr])/3.0 + ct['2007&'+attr]
    y='2011'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = (ct['2010&'+attr]+ct['2012&'+attr])/2.0
    y='2013'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = (ct['2012&'+attr]+ct['2014&'+attr])/2.0    
    y='2015'
    sub = [x for x in missing_cols if x[:4]==y]
    for c in sub:
        attr = c.split('&')[1]
        ct[c] = (ct['2014&'+attr]+ct['2016&'+attr])/2.0
    ct.shape

    for col in ct.columns[1:]:
        m = ct[col].mean()
        ct[col].fillna(m, inplace=True)

    stacked = ct.set_index(['country']).stack().reset_index()
    stacked['year'] = [x[0] for x in stacked.level_1.str.split('&')]
    stacked['score_type'] = [x[1] for x in stacked.level_1.str.split('&')]
    stacked.drop('level_1', axis=1, inplace=True)
    stacked.rename(columns={0:'score'}, inplace=True)
    unstacked = stacked.set_index(['country','year', 'score_type']).unstack().reset_index()
    unstacked.columns= ['country', 'year', 'customs' ,'infra', 'intl_ship'
                        , 'logistic_qlty', 'lpi', 'timeliness','track_trace']
    print(unstacked.shape)
    unstacked.head()

    lpi_all =unstacked
    #lpi_all.to_csv('lpi_2006-2016.csv')
    print("lpi describe: ", lpi_all.describe())
    return lpi_all

###### 3. Harmonize and join lpi and fsi with main data!  ######

def combine_logistics_and_stability_features(lpi_all, fsi_all):
    """ 
    Takes the two dataframes of the stability and logistics indices and combines
    Returns dataframe of combined
    """
    ### Harmonize the country_names
    print("\nCountry names to harmonize: ", lpi_all.country.unique(), fsi_all.country.unique())
    print("\nFSI Year Value counts: ",fsi_all.year.value_counts().sort_index())
    print("\nFSI Year Value counts: ",lpi_all.year.value_counts().sort_index())

    import os
    path = os.curdir+'\Data\Features\\'
    df_country = compare_columns(fsi_all, lpi_all,'country','country')
    df_country.to_csv(path+'fsi_lpi_country comps.csv')
    # After doing data cleaning in Excel, get the resulting map
    map1 = pd.ExcelFile(path+'fsi_lpi_country comps_map.xlsx').parse(sheetname='fsi-lpi-map')
    # Change lpi to match fsi
    lpi_all['country'].replace({x:y for x,y in zip(map1.lpi, map1.fsi)}, inplace=True)
    #df_random = compare_columns(fsi_all, lpi_all,'country','country')
    
    # Now combine lpi and fsi
    print(lpi_all.shape, fsi_all.shape)
    lpi_fsi_combined = pd.merge(fsi_all, lpi_all, how='left', left_on=['country', 'year'],
                               right_on=['country', 'year'])
    lpi_fsi_combined['year'] = pd.to_datetime(lpi_fsi_combined['year'])
    lpi_fsi_combined['year'] = [x.year for x in lpi_fsi_combined['year']]
    print(lpi_fsi_combined.shape)
    lpi_fsi_combined.describe()
    lpi_fsi_combined.to_csv(path+'lpi_fsi_combined.csv')
    return lpi_fsi_combined


###### 4. Origin Factory Address, Country, and Continent ##########

def generate_factory_location_features(main_data):
    """
    Takes in main data on supply chain, looksup addresses and locations using the googlemaps API
    Returns dataframe with factory, country and continent included 
    """
    ### Using factory to extract address and then, country maybe even distance and time to travel  
    factory = pd.DataFrame(main_data['factory'].unique(),columns=['factory'] )
    factory.head()
    # Generate list of corresponding addresses
    factory_address = []
    for f in factory['factory']:
        try:
            factory_address.append(getAddress(f))
        except IndexError:
            x = f.split()
            try:
                factory_address.append(getAddress(" ".join(x[-3:])))
            except IndexError:
                try:
                    factory_address.append(getAddress(" ".join(x[-2:])))
                except IndexError:
                    try:
                        factory_address.append(getAddress(" ".join(x[-1:])))
                    except IndexError:
                        factory_address.append(("IndexError","IndexError","IndexError"))
    # Check the length
    print('factory_address:', len(factory_address), "; factory:", len(factory))

    # Make sure addresses and factories have the same length
    assert len(factory_address) == len(factory), "Length Mismatch!"
    factory_address_df = pd.DataFrame(factory_address)
    # Concatenate the address to the factory
    factory['factory_address'], factory['origin_country'], factory['origin_continent'] = factory_address_df[0], \
    factory_address_df[1], factory_address_df[2]
    # How many factories were not identified/located?
    print(factory[factory['factory_address'] == 'IndexError']['factory_address'].value_counts(),'\n')
    # The unidentified factories..
    print('The unidentified factories...\n',list(factory[factory['factory_address'] == 'IndexError']['factory']))
    factory.origin_continent.value_counts()

    # Fix the missing ones..
    import os
    path = os.curdir + '\Data\Features\\'
    #fact_ref = pd.read_csv(path+'factory_map_premade.csv', encoding = "ISO-8859-1")
    fact_ref = pd.read_excel(path+'factory_map_premadeX.xlsx', encoding = "ISO-8859-1")
    factory_misfits_dict = {x:y for x, y in zip(fact_ref.name,fact_ref.factory_address) 
    if x in list(factory[factory['factory_address'] == 'IndexError']['factory'])}
    # Make the factory name an index of the factory df
    factory.index = factory['factory']
    try:
        factory['factory_address'] = factory['factory'].apply(
                lambda x: factory_misfits_dict.get(x, factory.loc[x]['factory_address']))
    except NameError:
        pass
    
    # Massage the data..
    # rename with better names
    factory['name'] = factory.index
    # drop redundant factory column
    factory.drop(['factory'], axis = 1, inplace=True)
    # intro numerical index to replace the other
    factory.index = range(len(factory))
    # Which indices have the error on continent and country?
    no_country_idx  = list(factory[factory.origin_continent=='IndexError']['factory_address'].index)
    # Make a country to continent map
    continent_dict = dict(zip(factory.origin_country, factory.origin_continent))

    # Update the missing specifc places for country and continent
    for i in no_country_idx:
        ctry = factory.iloc[i]['factory_address'].split()[-1]
        cntnt = getAddress(ctry)[2]
        factory.iloc[i]['origin_country'] = ctry
        factory.iloc[i]['origin_continent'] = cntnt
    import os
    path = os.curdir+'\Data\Features\\'
    factory.to_csv(path+"factory_country_continent.csv")
    print("factory shape: ",factory.shape)
    factory.head()

    # Check countries across two list of supply and demand
    orig_list = list(factory['origin_country'])
    dest_list = list(main_data.country)
    # Countries which are both origins and destinations
    print("Countries in both origin and destination: ",sorted([c for c in orig_list if c in dest_list]))
    # Drop dups
    factory = factory.drop_duplicates(keep='first')
    return factory

def add_factory_origin_features(main_data, factory): 
    return pd.merge(main_data, factory, how="left", left_on=['factory'], right_on=['name'])

def destination_and_origin_lpi_fsi_indicators(main_data, lpi_fsi_combined): # Must have origin factory/country first
    """
    Takes in main data on supply chain as well as combined lpi_fsi_data, 
    harmonizes the country names and returns 2 dfs, one for origin and another for 
    destination metrics for both stability and logistics indices
    """
    # First hamonize with main data
    #------------------------------------------------#
    dest = main_data.groupby(['country', 'del_date_scheduled_yr']).agg(['count', 'sum', 'mean'])['delayed']
    orig = main_data.groupby(['origin_country', 'del_date_scheduled_yr']).agg(['count', 'sum', 'mean'])['delayed']
    dest.reset_index(inplace=True)
    orig.reset_index(inplace=True)
    #orig['origin_country'] = orig.country
    #orig.drop('country', axis=1)
    print("Destination countries data: ",dest.shape
          ,"Unique countries: ",dest.country.unique()
          ,"Origin countries data: ", orig.shape
         ,"Unique countries: ", orig.origin_country.unique())

    compare_columns(dest,lpi_fsi_combined, 'country', 'country')

    lpi_fsi_combined['country'].replace({'Congo Democratic Republic':'Congo, DRC'
    ,"Cote d'Ivoire":"Côte d'Ivoire", 'Kyrgyz Republic':'Kyrgyzstan'}, inplace=True)

    compare_columns(dest, lpi_fsi_combined, 'country', 'country')
    compare_columns(orig, lpi_fsi_combined, 'origin_country', 'country')
    #------------------------------------------------#
    
    destination_metrics_by_year = pd.merge(dest,lpi_fsi_combined, how='left'
               , left_on=['country','del_date_scheduled_yr'], right_on=['country','year'])
    print(destination_metrics_by_year.columns,destination_metrics_by_year.describe(),
         destination_metrics_by_year.head())
    #destination_metrics_by_year.to_csv('destination_metrics_by_year.csv')

    origin_metrics_by_year = pd.merge(orig,lpi_fsi_combined, how='left'
               , left_on=['origin_country','del_date_scheduled_yr'], right_on=['country','year'])
    destination_metrics_by_year = destination_metrics_by_year.drop_duplicates(keep='first')
    origin_metrics_by_year = origin_metrics_by_year.drop_duplicates(keep='first')
    #print(origin_metrics_by_year.columns,origin_metrics_by_year.describe(),
    #     origin_metrics_by_year.head())
    #origin_metrics_by_year.to_csv('origin_metrics_by_year.csv')
    return destination_metrics_by_year, origin_metrics_by_year

def add_lpi_fsi_features(main_data, origin_metrics, destination_metrics): # Must Happen after they are generated!
    """
    Adds newly generated destination and origin country logistics and country fragility/peace features 
    to the main data.
    """
    # Prep the datarames
    orig, dest = origin_metrics, destination_metrics
    dest = dest.groupby(['dest_country', 'dest_del_date_scheduled_yr']).agg('mean')
    orig = orig.groupby(['orig_origin_country', 'orig_del_date_scheduled_yr']).agg('mean')
    dest, orig = dest.reset_index(), orig.reset_index()
    # Merge the data consecutively
    # Join the odd ones into dobject
    do1 = pd.merge(main_data,dest, how='left',left_on=['country', 'del_date_scheduled_yr']
             , right_on=['dest_country', 'dest_del_date_scheduled_yr'])
    do2 = pd.merge(do1, orig, how='left',left_on=['origin_country', 'del_date_scheduled_yr']
             , right_on=['orig_origin_country', 'orig_del_date_scheduled_yr'])
    main_data = do2.copy()    
    return main_data

def country_metrics_corr(data):
    """ 
    Takes in data with destination or origin metrics aggregated
    Retruns heatmap of correlations for visualizing. Compares against mean, sum and count
    """
    drop = [x for x in data.columns if ("sum" in x) or ("count" in x)]
    d=data.drop(drop, axis=1)
    f, ax = plt.subplots(1,1, figsize=(8,6))
    #print(d.corr())
    sns.heatmap(d.corr())


###############################################################################
        # -------- ML MODEL DEVELOPMENT AND REFINEMENT --------------#
###############################################################################
from sklearn.pipeline import TransformerMixin

class Dummifier(TransformerMixin):
    """
    Creates dummies from DataFrame, returns dataframe with first level of dummies dropped
    """
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return pd.get_dummies(X, drop_first=True)
    
class Labeler(TransformerMixin):
    """
    Creates labels for series, returns dataframe with 0,1
    """
    def fit(self, y, X=None):
        return self
    def transform(self, y):
        from sklearn.preprocessing import LabelBinarizer
        enc = LabelBinarizer()
        return enc.fit_transform(y)
    
# Extract feature importances into a df
def plot_feature_importances(fitted_estimator, X_train, n_features, show_plot=True):
    """
    Takes in an estimator RFClassifier which has already been "fitted" with feature importances, the full data
    Plots the top n_features importance. Returns full dataframe of importances by feature, and importance from n_features
    """
    features_clf = X_train.columns
    key_features_clf = pd.DataFrame(fitted_estimator.feature_importances_, features_clf)
    key_features_clf.reset_index(inplace=True)
    key_features_clf.columns=['feature', 'importance']
    dfresult = key_features_clf.set_index('feature').sort_values('importance', ascending=False)[:n_features]
    print("Total Importance of {} features: {}".format(n_features,dfresult.sum()))
    if show_plot:
        dfresult.plot(kind="barh", ylim=(0,0.5))
    return key_features_clf.set_index('feature').sort_values('importance', ascending=False)

def pca_results(good_data, pca, min_feature_influence=0.1):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    Visualizes the PCA results
    '''
    import numpy as np
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = good_data.keys())
    components.index = dimensions
    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance'])
    variance_ratios.index = dimensions
    # Create a bar plot visualization
    fig, ((ax0,ax1),(ax2,ax3)) = plt.subplots(
        nrows=2,ncols=2,figsize = (14,8))
    axes = [ax0,ax1,ax2,ax3]
    # Plot the feature weights as a function of the components  506/333
    for d in dimensions: # for each dimension
        i = dimensions.index(d)
        idx =abs(components.loc[d,:])>min_feature_influence # Select most influential features
        components.loc[d,:][idx].sort_values(ascending=False).plot(
            kind="barh",ax=axes[i], title="Important Featues for: "+d)  #plot on separate axis
        axes[i].set_xlabel("Feature Weights")
    # Return a concatenated DataFrame
    pd.concat([variance_ratios, components], axis = 1)

def train_test_oversample(X, y, test_size=0.35, use_smote=False):
    """
    Returns oversampled X and y dataframes depending on key word args.
    """
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split
    # Train-Test split
    X_tr_clf_res, X_ts_clf_res, y_tr_clf_res, y_ts_clf_res = train_test_split(
            X,y,test_size=test_size, random_state=121)

    smote = SMOTE(random_state=121, ratio = 1.0)
    print("XTrain, yTrain shapes BEFORE Oversampling: {} and {}".format(X_tr_clf_res.shape, y_tr_clf_res.shape))
    if use_smote: # do oversample technique
        X_tr_clf_res, y_tr_clf_res = smote.fit_sample(X_tr_clf_res, y_tr_clf_res)
    print("XTrain, yTrain shapes AFTER Oversampling: {} and {}".format(X_tr_clf_res.shape, y_tr_clf_res.shape))
    # Convert to dataframes for helper formulae
    X_tr_clf_res = pd.DataFrame(X_tr_clf_res, columns=X.columns)
    y_tr_clf_res = pd.DataFrame(y_tr_clf_res, columns=y.columns)
    print("Shape of XTrain: {} yTrain: {} XTest: {} yTest: {}".format(
        X_tr_clf_res.shape, X_ts_clf_res.shape, y_tr_clf_res.shape, y_ts_clf_res.shape))
    return X_tr_clf_res, X_ts_clf_res, y_tr_clf_res, y_ts_clf_res


def train_test_conditional(X, y, df_true_pos, ddate, delayed):
    """
    Takes several datarames, output variable and return conditinal train test split sets.
    """
    # Set the data arrays right..train test split
    print("Now getting train test splits for regression...")
    X_tr_reg_all = X.loc[delayed[delayed==1].dropna().index.tolist(),:]
    X_tr_reg = X_tr_reg_all.drop(df_true_pos.index.tolist(),axis=0)

    y_tr_reg_all = ddate.loc[delayed[delayed==1].dropna().index.tolist(),['delivery_delay_time']]
    y_tr_reg_tp = y_tr_reg_all.drop(df_true_pos.index.tolist(),axis=0)
    y_tr_reg = y_tr_reg_tp.delivery_delay_time.dt.days

    # Do the train test split business
    X_ts_reg = X.loc[df_true_pos.index.tolist(),:]
    y_ts_reg = ddate.loc[df_true_pos.index.tolist(),['delivery_delay_time']]['delivery_delay_time'].dt.days
    print("Shapes:\n {}\n,{}\n,{}\n,{}".format(
                    X_tr_reg.shape, X_ts_reg.shape, y_tr_reg.shape, y_ts_reg.shape))
    return X_tr_reg, X_ts_reg, y_tr_reg, y_ts_reg

def fit_and_generate_true_positives(estimator, X_tr_clf_res, X_ts_clf_res, y_tr_clf_res, y_ts_clf_res):
    """
    ?Returns dataframe of true positives and dataframe of actual vs. pred
    """
    # Fit the regression to the data, training!
    clf = estimator
    clf.fit(X_tr_clf_res, y_tr_clf_res)
    # Predicts test values, Accuracy and error calcs
    y_pred_clf = clf.predict(X_ts_clf_res)
    # Make dataframe of delayed vs. predicted
    df_pred=pd.DataFrame(y_ts_clf_res, columns=['delayed']); df_pred['pred']= y_pred_clf
    # Put all the corectly identified delayed items into one dataframe 
    # To be used in the prediction of length/extent of delay 
    df_true_pos = df_pred[(df_pred.delayed==1) & (df_pred.delayed==df_pred.pred)]
    print("Shape of true positive df: ", 
              df_true_pos.shape,"Number of 1's in true positive df:", df_true_pos.sum())
    return df_true_pos, df_pred

# The model selection function from my helper functions
def model_selection(X_train, X_test, y_train, y_test, estimator, alg_type):
    """
    Takes train and test data sets for both features and target plus an estimator and 
    returns f1_score or a tuple of r2 and RMSE. So be careful which alg_type you want.
    """
    # Scoring functions and some estimator built in container
    from sklearn.metrics import f1_score, r2_score, mean_squared_error
    from sklearn.pipeline import Pipeline#, preprocessing
    from sklearn import preprocessing
    import numpy as np

    model = Pipeline([ #('label_encoding', EncodeCategorical(X.keys())),
         #('one_hot_encoder', OneHotEncoder()),
         ('estimator', estimator)])    
    if alg_type == 'clf':
        # Instantiate the classification model and visualizer
        y_train = preprocessing.LabelEncoder().fit_transform(y_train.values.ravel())
        y_test = preprocessing.LabelEncoder().fit_transform(y_test.values.ravel())
        model.fit(X_train, y_train)
        expected  = y_test
        predicted = model.predict(X_test)
        # Compute and return the F1 score (the harmonic mean of precision and recall)
        return (f1_score(expected, predicted))
    elif alg_type == 'reg':
        y_train = y_train.values.ravel()
        y_test = y_test.values.ravel()
        model.fit(X_train, y_train)
        expected  = y_test
        predicted = model.predict(X_test)
        # Compute and return the R2 and RMSE metrics
        r = (r2_score(expected, predicted), np.sqrt(mean_squared_error(expected, predicted)))
        return r

def visual_model_selection(X_train, X_test, y_train, y_test, estimator, show_plot=True):
    """
    Takes train and test data sets for both features and target plus an estimator and 
    returns a visual classification report.
    """ 
    from sklearn.pipeline import Pipeline 
    from yellowbrick.classifier import ClassificationReport
    #y_train = preprocessing.LabelEncoder().fit_transform(y_train.values.ravel())
    #y_test = preprocessing.LabelEncoder().fit_transform(y_test.values.ravel())
        
    model = Pipeline([('estimator', estimator)])

    # Instantiate the classification model and visualizer
    visualizer = ClassificationReport(model, classes=['on-time', 'delayed'])
    visualizer.fit(X_train, y_train)
    visualizer.score(X_test, y_test)
    visualizer.poof()
    return visualizer.scores

def run_combined_classify_regress_model(data, ddate, delayed,classifier, regressor
                                        , test_size=0.35, use_smote=False, show_plot=False):
    """
    Combined model which classifies and then does regression to find length of delay. 
    Plots several useful metrics for classification and regression.
    Saves the predicted vs. actual and true positive preditions to Data\Results folder
    Returns key dataframes for analysis
        df_pred_fin ----------> final predictions on the test observations 
        df_true_pos_fin ------> true postive predictions
        d_feat_imp_clf_fin ---> feature importances for the classification model
        d_rsq_fin ------------> regression scores R2 and RMSE
        clf_object -------> is the classification prediction object
    """ 
    #-------------------Classification----------------#
    # Get the train test splits # Use Oversampling
    X_tr_clf_fin,X_ts_clf_fin,y_tr_clf_fin,y_ts_clf_fin = train_test_oversample(data, delayed
                                                            , test_size=test_size, use_smote=use_smote)
    # Instatiatiate the models
    final_estimator = classifier
    print("\n----\n")
    # Fit the regression to the data and train
    final_clf = final_estimator().fit(X_tr_clf_fin, y_tr_clf_fin)
    try:
        d_feat_imp_clf_fin = plot_feature_importances(final_clf, X_tr_clf_fin, 30,show_plot=show_plot)
    except AttributeError:
        print("Classifier has no feature importance attributes")
        d_feat_imp_clf_fin = pd.DataFrame([[]])
    # Predict and plot F1-Scores, Precision and Recall
    plt.subplots(figsize=(6,5))
    clf_scores = visual_model_selection(X_tr_clf_fin, X_ts_clf_fin, y_tr_clf_fin
                                        , y_ts_clf_fin, final_clf, show_plot=show_plot)
    df_true_pos_fin, df_pred_fin = fit_and_generate_true_positives(final_estimator, X_tr_clf_fin
                                                                 , X_ts_clf_fin, y_tr_clf_fin, y_ts_clf_fin)
    # Save to disk
    path = os.curdir+"\Data\Results\\"
    df_pred_fin.to_csv(path+'classifier_final_predicted.csv')
    df_true_pos_fin.to_csv(path+'classifier_final_true_positives.csv') # Save to data folders
    
    #-------------------Regression----------------#
    reg_estimator_fin = regressor
    # Get training and test sets...
    X_tr_reg_fin, X_ts_reg_fin, y_tr_reg_fin, y_ts_reg_fin = train_test_conditional(data,
                                                                delayed, df_true_pos_fin, ddate, delayed)
    rsq = []
    rsq.append(model_selection(X_tr_reg_fin, X_ts_reg_fin
                                   , y_tr_reg_fin, y_ts_reg_fin, reg_estimator_fin(), 'reg'))
    d_rsq_fin = pd.DataFrame(rsq, columns=['r2', 'rmse'])
    return df_pred_fin, df_true_pos_fin,  d_feat_imp_clf_fin, clf_scores, d_rsq_fin 

def run_combined_classify_regress_model_prefit(data, ddate, delayed,classifier, regressor
                                        , test_size=0.35, use_smote=False, show_plot=False):
    """
    Combined model which classifies and then does regression to find length of delay. 
    Plots several useful metrics for classification and regression.
    Saves the predicted vs. actual and true positive preditions to Data\Results folder
    Returns key dataframes for analysis
        df_pred_fin ----------> final predictions on the test observations 
        df_true_pos_fin ------> true postive predictions
        d_feat_imp_clf_fin ---> feature importances for the classification model
        d_rsq_fin ------------> regression scores R2 and RMSE
        clf_object -------> is the classification prediction object
    """ 
    from sklearn.metrics import r2_score, mean_squared_error, classification_report, confusion_matrix
    import numpy as np
    #-------------------Classification----------------#
    # Get the train test splits # Use Oversampling
    X_tr_clf_fin,X_ts_clf_fin,y_tr_clf_fin,y_ts_clf_fin = train_test_oversample(data, delayed
                                                            , test_size=test_size, use_smote=use_smote)
    # Instatiatiate the models
    final_estimator = classifier
    print("\n----\n")
    # Fit the regression to the data and train
    final_clf = final_estimator.fit(X_tr_clf_fin, y_tr_clf_fin)
    try:
        d_feat_imp_clf_fin = plot_feature_importances(final_clf, X_tr_clf_fin, 30,show_plot=show_plot)
    except AttributeError:
        print("Classifier has no feature importance attributes")
        d_feat_imp_clf_fin = pd.DataFrame([[]])
    # Predict and plot F1-Scores, Precision and Recall    
    clfreport, cmatrix=[m(y_ts_clf_fin,final_estimator.predict(X_ts_clf_fin)) 
                        for m in [classification_report, confusion_matrix] ]
    # Make dataframe of delayed vs. predicted
    df_pred_fin=pd.DataFrame(y_ts_clf_fin, columns=['delayed'])
    df_pred_fin['pred']= final_estimator.predict(X_ts_clf_fin)
    # Make DataFrame of True Positives for regression prediction (length/extent of delay) 
    df_true_pos_fin = df_pred_fin[(df_pred_fin.delayed==1) & (df_pred_fin.delayed==df_pred_fin.pred)]
    # Save to disk
    path = os.curdir+"\Data\Results\\"
    df_pred_fin.to_csv(path+'classifier_final_predicted.csv')
    df_true_pos_fin.to_csv(path+'classifier_final_true_positives.csv') # Save to data folders
    #-------------------Regression----------------#
    reg_estimator_fin = regressor
    # Get training and test sets...
    X_tr_reg_fin, X_ts_reg_fin, y_tr_reg_fin, y_ts_reg_fin = train_test_conditional(data,
                                                                delayed, df_true_pos_fin, ddate, delayed)
    reg_estimator_fin.fit(X_tr_reg_fin,y_tr_reg_fin)
    r2= r2_score(y_ts_reg_fin,reg_estimator_fin.predict(X_ts_reg_fin))
    rmse = np.sqrt(mean_squared_error(y_ts_reg_fin,reg_estimator_fin.predict(X_ts_reg_fin)))
    #d_rsq_fin = pd.DataFrame([r2,rmse], columns=['r2', 'rmse'])
    return df_pred_fin, df_true_pos_fin,  d_feat_imp_clf_fin, clfreport, cmatrix, r2, rmse 