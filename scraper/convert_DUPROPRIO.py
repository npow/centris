# -*- coding: utf-8 -*-
# vim: tabstop=2 expandtab shiftwidth=2 softtabstop=2
import csv
import pandas as pd
import re

def parseFloat(s):
  s = re.sub("[^\d\.]", "", s)
  if len(s) == 0:
    return ''
  return float(s)

def parseInt(s):
  s = re.sub("[^\d\.]", "", s)
  if len(s) == 0:
    return ''
  return int(s)

INPUT_FILE = 'data/archive/hist_DUPROPRIO.csv'
OUTPUT_FILE = 'data/hist_DUPROPRIO_v2.csv'
listings = list(csv.DictReader(open(INPUT_FILE), delimiter=',', quotechar='"'))

L = []
keys = [
  'NumberBedrooms', 'NumberBathrooms', 'LivingArea', 'LotSize', 'Category', 'AskingPrice', 'PriceSold',
  'SaleYYYY', 'SaleMM', 'SaleDD', 'SaleYYYYMMDD', 'DaysOnMarket', 'Address', 'Borough'
]
for i, listing in enumerate(listings):
  info = {}
  for key in keys:
    info[key] = ''
  listing['Info'] = re.sub('\xc2\xa0', ' ', listing['Info'])
  listing['Info'] = re.sub('\xef\xbf\xbd', '', listing['Info'])
  if listing['Info'] == '':
    continue

  m = re.match('.*([\d]+) bedroom\(s\).*', listing['Info'])
  if m:
    info['NumberBedrooms'] = parseInt(m.groups()[0])

  m = re.match('.*([\d]+) bathroom\(s\).*', listing['Info'])
  if m:
    info['NumberBathrooms'] = parseInt(m.groups()[0])

  m = re.match('.*Living space  ([\d,]+).*', listing['Info'])
  if m:
    info['LivingArea'] = parseInt(m.groups()[0])

  m = re.match('.*Lot ([\d,]+)(.*)', listing['Info'])
  if m:
    info['LotSize'] = parseInt(m.groups()[0])

  info['Category'] = listing['Category']
  info['AskingPrice'] = parseFloat(listing['AskingPrice'])
  info['PriceSold'] = parseFloat(listing['PriceSold'])

  m = re.match('.*on (.*)', listing['SaleDate'])
  if m:
    dateSold = m.groups()[0]
    info['SaleYYYY'] = parseInt(dateSold.split('-')[0])
    info['SaleMM'] = parseInt(dateSold.split('-')[1])
    info['SaleDD'] = parseInt(dateSold.split('-')[2])
    info['SaleYYYYMMDD'] = info['SaleYYYY']*10000 + info['SaleMM']*100 + info['SaleDD']

    daysOnMarket = 0
    m = re.match('Sold in ([\d]+) (Days|day).*', listing['SaleDate'])
    if m:
      daysOnMarket += parseInt(m.groups()[0])
    m = re.match('Sold in ([\d]+) (Month|months).*', listing['SaleDate'])
    if m:
      daysOnMarket += parseInt(m.groups()[0]) * 30
    info['DaysOnMarket'] = daysOnMarket

  if listing['Address'].find('\\n') == -1:
    continue
  info['Borough'] = listing['Address'].split('\\n')[1].strip()
  info['Address'] = listing['Address'].split('\\n')[0].strip() + ' Montreal'

  L.append(info)


with open(OUTPUT_FILE, 'w') as f:
  fieldnames = L[0].keys()
  writer = csv.DictWriter(f, fieldnames=fieldnames)
  writer.writeheader()
  for listing in L:
    writer.writerow(listing)
