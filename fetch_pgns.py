import urllib.request
import datetime
import os

def fetch_games_last_year(username, output_file="my_games.pgn"):
    """
    Fetches the PGNs of all chess games played by a specific user in the last year
    using the chess.com API, and saves them to a file.
    """
    # Chess.com API requires a descriptive User-Agent header
    headers = {
        'User-Agent': f'Data extraction script for user {username}'
    }
    
    # Calculate the last 12 months
    today = datetime.date.today()
    current_year = today.year
    current_month = today.month
    
    months = []
    for _ in range(12):
        months.append((current_year, current_month))
        current_month -= 1
        if current_month == 0:
            current_month = 12
            current_year -= 1
            
    months.reverse() # Order from oldest to newest

    print(f"Fetching games for {username} from {months[0][0]}-{months[0][1]:02d} to {months[-1][0]}-{months[-1][1]:02d}...")

    # Open the output file in write mode
    with open(output_file, "w") as out:
        games_found = False
        for year, month in months:
            # Chess.com API endpoint for downloading a month's games as PGN
            url = f"https://api.chess.com/pub/player/{username}/games/{year}/{month:02d}/pgn"
            print(f"Fetching {year}-{month:02d}...")
            
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    pgn_data = response.read().decode('utf-8')
                    if pgn_data.strip():
                        out.write(pgn_data)
                        out.write("\n")
                        games_found = True
            except urllib.error.HTTPError as e:
                # 404 means no games played that month, skip silently or log
                if e.code == 404:
                    print(f"  -> No games found for {year}-{month:02d}.")
                else:
                    print(f"  -> Error fetching data for {year}-{month:02d}: {e}")
            except Exception as e:
                print(f"  -> Error fetching data for {year}-{month:02d}: {e}")
                
    if games_found:
        print(f"\nSuccessfully downloaded games and saved to {output_file}.")
    else:
        print(f"\nNo games found for {username} in the last 12 months.")

if __name__ == "__main__":
    # Replace with your actual Chess.com username
    USERNAME = "SRK6776" 
    OUTPUT_FILE = "my_games.pgn"
    
    fetch_games_last_year(USERNAME, OUTPUT_FILE)
