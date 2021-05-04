# reddit scraper, by subreddit
# results will be saved to csv file named <subreddit>.csv

import csv
import os
import requests

reddit_fields = ['subreddit', 'url']
headers = {'User-agent': 'nlp scraper 0.0.1'}
sort = '?sort=top&t=all&limit=100'

whitelist = []

# read in whitelist
with open('whitelist.txt') as f:
    whitelist = f.readlines()
whitelist = [line.strip() for line in whitelist]


def main():
    REDDIT_BASE = 'https://old.reddit.com/r/'
    after = None

    subreddits = input('Enter subreddits to scrape: ').split(' ')

    for s in subreddits:
        for _ in range(5):
            page = f'&after={after}' if after else ''
            r = requests.get(
                f'{REDDIT_BASE}{s}/top/.json{sort}{page}', headers=headers)
            if r.ok:
                res = r.json()['data']
                posts = process_reddit_response(res['children'])
                write_subreddit_to_csv(s, posts)
                after = res['after']


def process_reddit_response(sub_response):
    posts = []
    for post in sub_response:
        p = post['data']
        if p and not p['stickied'] and accepted(p['url']):
            # get subreddit name and url
            d = {x: p[x] for x in reddit_fields}
            # check if post's url is on whitelist
            posts.append(d)
    return posts


def accepted(url):
    accept = True
    blacklist = ['i.redd.it', 'imgur', 'reddit',
                 'streamable', 'facebook.com',
                 'twitter.com', 'instagram.com', 'redd.it',
                 '.gif', '.jpg', '.mp4', '.png', 'youtube.com',
                 'youtu.be', 'decisiondeskhq.com', 'duckduckgo']

    for b in blacklist:
        accept = accept and b not in url

    return accept


def write_subreddit_to_csv(subreddit, posts):
    filename = f'{subreddit}.csv'
    mode = 'a' if os.path.exists(filename) else 'w'
    with open(filename, mode=mode) as f:
        writer = csv.writer(f)
        if mode == 'w':
            writer.writerow(reddit_fields)
        for p in posts:
            row = [p[x] for x in reddit_fields]
            writer.writerow(row)


def url_in_whitelist(url):
    return any(sitename in url for sitename in whitelist)


if __name__ == '__main__':
    main()
