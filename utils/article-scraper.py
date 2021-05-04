import csv
from newsplease import NewsPlease


def main():
    csv_list = input('Enter CSVs to scrape: ').split(' ')

    with open('scraped.csv', 'w') as f:
        writer = csv.writer(f)

        for c in csv_list:
            with open(c, 'r') as source:
                source.readline()
                count = 1
                for line in source:
                    sub, url = line.split(',')
                    print(f'Scraping article {count}: {url}')
                    count += 1
                    try:
                        a = NewsPlease.from_url(url, timeout=10)
                    except Exception:
                        print(url)
                        continue
                    if a.maintext is not None:
                        writer.writerow([sub, url, a.maintext])


if __name__ == '__main__':
    main()
