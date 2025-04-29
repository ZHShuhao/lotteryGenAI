import http.client
import json

def fetch_power_ball_data():
    conn = http.client.HTTPSConnection("powerball.p.rapidapi.com")
    headers = {
         'x-rapidapi-key': "b5b78de802msh45b4a624c9a36b0p196f8djsn0b3ec34ef836",
         'x-rapidapi-host': "powerball.p.rapidapi.com"
    }
    conn.request("GET", "/", headers=headers)
    res = conn.getresponse()
    data = res.read()

    # Parse JSON and return the desired fields
    json_data = json.loads(data.decode("utf-8"))
    draws = json_data["data"]
    extracted_data = [
        {
            "DrawingDate": draw["DrawingDate"],
            "Number1": draw["FirstNumber"],
            "Number2": draw["SecondNumber"],
            "Number3": draw["ThirdNumber"],
            "Number4": draw["FourthNumber"],
            "Number5": draw["FifthNumber"],
            "PowerBall": draw["PowerBall"],
            "Jackpot": draw["Jackpot"],
            "EstimatedCashValue": draw["EstimatedCashValue"]
        }
        for draw in draws
    ]
    return extracted_data
