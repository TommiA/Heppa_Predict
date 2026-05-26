import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import time
import random
import csv
import os

BASE = "https://heppa.hippos.fi"
CALENDAR_URL = f"{BASE}/heppa/app?page=racing%2FRaceCalendarEarlierYear&service=external"
#CALENDAR_URL = f"{BASE}/heppa/app?page=racing%2FRaceCalendarEarliestYear&service=external"

OUTPUT_CSV = "heppa_all_results.csv"

def get_meeting_urls(soup):
    links = soup.select("a[href]")
    return [urljoin(BASE, a["href"]) for a in links if "page=racing%2FRaceResults&service" in a["href"]]

def get_calendar_next(soup):
    nxt = soup.find("a", string=re.compile(r"Seuraava|Next"))
    return urljoin(BASE, nxt["href"]) if nxt else None

def extract_value(label_text, soup):
    row = soup.find("td", string=label_text)
    if row:
        value_td = row.find_next_sibling("td")
        if value_td:
            return value_td.text.strip()
    return None

def get_meeting_info(soup):
    span = soup.find("span", class_="page_header")
    text = span.get_text(strip=True) if span else ""
    m = re.search(r"Tulokset,\s+(\d{1,2}\.\d{1,2}\.\d{4})\s+(\d{1,2}:\d{2})\s+(.+)", text)
    temperature = extract_value("Lämpötila", soup)
    track_condition = extract_value("Radan kunto", soup)
    return (m.group(3), m.group(1), m.group(2), temperature, track_condition) if m else (None, None, None, temperature, track_condition)

def get_heat_number(soup):
    h3 = soup.select_one("div.datablock > h3")
    if h3:
        m = re.match(r"(\d+)\.\s*lähtö", h3.get_text(strip=True))
        if m:
            return m.group(1), h3.get_text()
    return None, None

def get_heat_urls(soup):
    heat_urls = []
    for a in soup.select("a[href]"):
        href = a["href"]
        if (
            "page=racing%2FRaceResults" in href
            and "sp=CC" in href
            and href.count("sp=") == 3
        ):
            heat_urls.append(urljoin(BASE, href))
    return sorted(heat_urls, key=lambda url: int(re.search(r"sp=CC(\d+)", url).group(1)))

def parse_heat_page(url, location, date, start_time, temperature, track_condition):
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    
    results = []
    heat_num, heat_info = get_heat_number(soup)

    for row in soup.select("table.raceResultTable tr")[1:]:
        cols = row.find_all("td")
        if not cols or len(cols) < 8:
            continue
        
        placing = cols[0].get_text(strip=True)
        number = cols[1].get_text(strip=True)

        horse_driver_cell = cols[2]
        links = horse_driver_cell.find_all("a")
        horse_name = links[0].get_text(strip=True) if len(links) > 0 else ""
        driver_name = links[1].get_text(strip=True) if len(links) > 1 else ""

        time_val = cols[3].get_text(strip=True)
        notes = cols[4].get_text(strip=True)
        odds = cols[5].get_text(strip=True)
        prize = cols[6].get_text(strip=True)
        distance = cols[7].get_text(strip=True)

        result = {
            "date": date,
            "start_time": start_time,
            "location": location,
            "heat_num": heat_num,
            "placing": placing,
            "number": number,
            "horse": horse_name,
            "driver": driver_name,
            "time": time_val,
            "note": notes,
            "odds": odds,
            "prize": prize,
            "distance": distance,
            "temperature": temperature,
            "track_condition": track_condition,
            "heat_info": heat_info
        }

        results.append(result)

    return results

def load_existing_dates(csv_path):
    existing_dates = set()
    if not os.path.exists(csv_path):
        return existing_dates
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            existing_dates.add((row["date"], row["location"]))
    return existing_dates

def main():
    cal_url = CALENDAR_URL
    all_meetings = []
    existing_dates = load_existing_dates(OUTPUT_CSV)

    #a['datetime']=pd.to_datetime(a.date, dayfirst=True)

    fieldnames = [
        "location", "date", "start_time", "heat_num",
        "placing", "number", "horse", "driver",
        "time", "odds", "distance", "note", "prize", "temperature", "track_condition", "heat_info"
    ]

    write_header = not os.path.exists(OUTPUT_CSV)

    with open(OUTPUT_CSV, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        while cal_url:
            print(f"Fetching calendar: {cal_url}")
            soup = BeautifulSoup(requests.get(cal_url).text, "html.parser")
            meetings = get_meeting_urls(soup)
            all_meetings.extend(meetings)
            cal_url = get_calendar_next(soup)
            time.sleep(random.uniform(1, 2))

        for meet_url in all_meetings:
            print(f"\nMeeting: {meet_url}")
            meet_resp = requests.get(meet_url)
            meet_soup = BeautifulSoup(meet_resp.text, "html.parser")
            location, date, start_time, temperature, track_condition = get_meeting_info(meet_soup)
            
            if not date or not location:
                print("  Skipping: could not extract date or location.")
                continue
            
            if (date, location) in existing_dates:
                print(f"  Skipping {date} at {location} (already in CSV)")
                continue

            print(f"Processing {date} {location} @ {start_time}")
            heat_urls = get_heat_urls(meet_soup)

            for heat_url in heat_urls:
                print(f"  Heat: {heat_url}")
                rows = parse_heat_page(heat_url, location, date, start_time, temperature, track_condition)
                for row in rows:
                    writer.writerow(row)
                time.sleep(random.uniform(1, 2))

if __name__ == "__main__":
    main()
