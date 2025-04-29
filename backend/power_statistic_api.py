import http.client
import json

def fetch_power_ball_statistic_data():

    conn = http.client.HTTPSConnection("powerball.p.rapidapi.com")

    headers = {
        'x-rapidapi-key': "b5b78de802msh45b4a624c9a36b0p196f8djsn0b3ec34ef836",
        'x-rapidapi-host': "powerball.p.rapidapi.com"
    }

    conn.request("GET", "/stats", headers=headers)

    res = conn.getresponse()
    data = res.read()

    # print(data.decode("utf-8"))
    # Parse JSON
    json_data = json.loads(data.decode("utf-8"))
    data_section = json_data.get("data",{})
    print(json.dumps(data_section, indent=4))

    # Extract `whiteballoccurrences` and `megaBalloccurrences`
    whiteball_occurrences = data_section.get("whiteballoccurrences", {})
    power_ball_occurrences = data_section.get("powerballoccurrences", {})

    extracted_data = {
        "whiteballoccurrences": whiteball_occurrences,
        "powerballoccurrences": power_ball_occurrences
    }

    return extracted_data