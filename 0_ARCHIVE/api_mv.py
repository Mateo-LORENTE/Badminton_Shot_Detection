import requests
import json
import pandas as pd

# authentification à l'API
ffl = "https://tagging.meta-video.com/insep_api"
ffl_core = "https://tagging.meta-video.com/api"
apikey = "012568e6-5120-487e-8826-d75520c94772"

# Récupérer les infos sur les matchs et leur session
def get_matchs_report(type='dict'):
    payload = json.dumps({"apiKey": apikey, "request": "get_match_report"})
    get_match_report = requests.get(ffl, data=payload)
    match_report = json.loads(get_match_report.text)
    match_report = match_report['data']
    if type == 'df' or type == 'dataframe' or type == 'DataFrame':
        return pd.DataFrame(match_report)
    else:
        return match_report

# Récupérer les infos sur la vidéo
def get_video_info(videoId, account):
    payload = json.dumps({"apiKey": apikey, "request": "getVideoInfo", "account": account, "data": {"id": int(videoId)}})
    get_video_info = requests.get(ffl_core, data=payload)
    videoinfo_report = json.loads(get_video_info.text)
    videoinfo_report = videoinfo_report['data']
    return videoinfo_report

# Récupérer les ids des matchs
def get_matchs_ids():
    match_report = get_matchs_report('df')
    return match_report[['id', 'name', 'analyse']]

# Récupérer les tags pour un match donné
def get_tags_report(match_id, type='dict'):
    payload = json.dumps({"apiKey": apikey, "request": "get_tag_report", "data": {"matchId": int(match_id)}})
    get_tags_report = requests.get(ffl, data=payload)
    tag_report = json.loads(get_tags_report.text)
    tag_report = tag_report['data']
    if type == 'df' or type == 'dataframe' or type == 'DataFrame':
        return pd.DataFrame(tag_report)
    else:
        return tag_report

# Récupérer l'URL d'une vidéo
def get_video_url(videoId):
    video_info = get_video_info(videoId)
    return video_info['replayUrl']

# Récupérer le contenu de la table tag
def get_t_content(type='dict'):
    payload = json.dumps({"apiKey": apikey, "request": "get_table_content", "data": {"tableName": "tag"}})
    get_t_content = requests.get(ffl, data=payload)
    t_content = json.loads(get_t_content.text)
    t_content = t_content['data']
    if type == 'df' or type == 'dataframe' or type == 'DataFrame':
        return pd.DataFrame(t_content)
    else:
        return t_content

# Récupérer les infos sur les athlètes
def get_athletes(type='dict'):
    payload = json.dumps({"apiKey": apikey, "request": "get_table_content", "data": {"tableName": "athlete"}})
    get_athletes = requests.get(ffl, data=payload)
    athletes_report = json.loads(get_athletes.text)
    athletes_report = athletes_report['data']
    if type == 'df' or type == 'dataframe' or type == 'DataFrame':
        return pd.DataFrame(athletes_report)
    else:
        return athletes_report

# Récupérer les coordonnées
def get_coordinates(matchId):
    payload = json.dumps({"apiKey": apikey, "request": "get_junction_table", "data": {"table": "coordinates", "matchId": int(matchId)}})
    get_coordinates = requests.get(ffl, data=payload)
    coordinates_report = json.loads(get_coordinates.text)
    coordinates_report = pd.DataFrame(coordinates_report['data'])
    return coordinates_report

# Récupérer les infos sur les équipes
def get_teams_report(matchId):
    payload = json.dumps({"apiKey": apikey, "request": "get_team_report", "data": {"matchId": int(matchId)}})
    get_team_report = requests.get(ffl, data=payload)
    team_report = json.loads(get_team_report.text)
    team_report = pd.DataFrame(team_report['data'])
    return team_report


"""
# Fichier qui défini toutes les fonctions pour communiquer avec l'API Meta-Video
import requests
import json
import pandas as pd

# authentification à l'API
ffl = "https://tagging.meta-video.com/insep_api"
login = "FFBAD_Double"                                                             ## login
#pwd = "834c6c4e9c1ff3bd1ce5fbfceb4066889de95259faf95091b1bce40742ef5b85"  ## password encrypté sha256
pwd = "20f23f3b740b127c25ead20ea934ca6bc848c008b1c440d13c11773a43cdfb39"  ## password encrypté sha256

#Récupérer les infos sur les matchs et leur session
def get_matchs_report(type='dict'):
        
    get_matchs_report = requests.get(ffl,params={
            "login":login,
            "pwd":pwd,
            "request":"get_match_report",
            "account":"ffbad_double",
            })

    matchs_report=json.loads(get_matchs_report.text)
    matchs_report=matchs_report['data']

    if type=='df' or type=='dataframe' or type=='Dataframe' or type=='DataFrame':
        return pd.DataFrame(matchs_report)
    else:
        return matchs_report
    
# Récupérer les ids des matchs
def get_matchs_ids():
    match_report = get_matchs_report('df')
    return match_report[['id','name','analyse']]

# Récupérer les tags pour un match donné
def get_tags_report(match_id,type='dict'):
    get_tags_report = requests.get(ffl,params={
            "login":login,
            "pwd":pwd,
            "request":"get_tag_report",
            "account":"ffbad_double",
            'matchId':str(match_id)
            })
    
    tags_report=json.loads(get_tags_report.text)
    tags_report=tags_report['data']

    if type=='df' or type=='dataframe' or type=='Dataframe' or type=='DataFrame':
        return pd.DataFrame(tags_report)
    else:
        return tags_report
    
# Récupérer l'URL d'une vidéo: 
def get_video_url(videoId):
    get_video_info = requests.get(ffl,params={
        "login":login,
        "pwd":pwd,
        "request":"get_video_info",
        "account":"FFBAD",
        "id":str(videoId)
        })

    video_info=json.loads(get_video_info.text)
    video_info=video_info['data']
    
    return video_info['replayUrl']
    
# Récupérer le contenu de la table tag
def get_t_content(type='dict'):
    get_t_content = requests.get(ffl,params={
            "login":login,
            "pwd":pwd,
            "request":"get_table_content",
            "account":"FFBAD",
            "tableName":'tag'
            })
    t_content=json.loads(get_t_content.text)
    t_content=t_content['data']

    for tag_type in t_content:
        del tag_type['del']

    if type=='df' or type=='df''dataframe' or type=='df''Dataframe' or type=='df''DataFrame':
        return pd.DataFrame(t_content)
    else:
        return t_content
    
# Récupérer les infos sur les athlètes
def get_athletes(type='dict'):
    get_athletes = requests.get(ffl,params={
        "login":login,
        "pwd":pwd,
        "request":"get_athletes",
        "account":"FFBAD",
        })
    athletes_report=json.loads(get_athletes.text)
    athletes_report=athletes_report['data']

    if type=='df' or type=='df''dataframe' or type=='df''Dataframe' or type=='df''DataFrame':
        return pd.DataFrame(athletes_report)
    else:
        return athletes_report
    
# On regroupe la session et la liste des tags
def create_update(session,tags):
    update={}
    update['session']=session
    update['tags']=tags
    return update

# push sur l'API Metavideo
def upload_tags(update): 
    params=json.JSONEncoder().encode({"login":login,
            "pwd":pwd,
            "action":"save_sequencing_data",
            "account":"FFBAD",
            "data": update
            })
    
    return requests.post(ffl,data=params,json=params)
"""