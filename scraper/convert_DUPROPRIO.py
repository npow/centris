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

INPUT_FILE = 'data/hist_DUPROPRIO.csv'
OUTPUT_FILE = 'data/hist_DUPROPRIO_v2.csv'
listings = list(csv.DictReader(open(INPUT_FILE), delimiter=',', quotechar='"'))

L = []
for i, listing in enumerate(listings):
  info = {}
  listing['Info'] = re.sub('\xc2\xa0', ' ', listing['Info'])
  listing['Info'] = re.sub('\xef\xbf\xbd', '', listing['Info'])
  if listing['Info'] == '':
    continue
  m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\), Living space  ([\d,]+) ft, Lot ([\d,]+)(.*)', listing['Info'])
  if m:
    info['NumberBedrooms'] = parseInt(m.groups()[0])
    info['NumberBathrooms'] = parseInt(m.groups()[1])
    info['LivingArea'] = parseFloat(m.groups()[2])
    info['LotSize'] = parseFloat(m.groups()[3])
  else:
    m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\), Living space  ([\d,]+) ft', listing['Info'])
    if m:
      info['NumberBedrooms'] = parseInt(m.groups()[0])
      info['NumberBathrooms'] = parseInt(m.groups()[1])
      info['LivingArea'] = parseFloat(m.groups()[2])
      info['LotSize'] = ''
    else:
      m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\), Lot ([\d,]+) ft', listing['Info'])
      if m:
        info['NumberBedrooms'] = parseInt(m.groups()[0])
        info['NumberBathrooms'] = parseInt(m.groups()[1])
        info['LivingArea'] = ''
        info['LotSize'] = parseFloat(m.groups()[2])
      else:
        m = re.match('([\d]+) bedroom\(s\), Living space  ([\d,]+) ft, Lot ([\d,]+)(.*)', listing['Info'])
        if m:
          info['NumberBathrooms'] = ''
          info['NumberBedrooms'] = parseInt(m.groups()[0])
          info['LivingArea'] = parseFloat(m.groups()[1])
          info['LotSize'] = parseFloat(m.groups()[2])
        else:
          m = re.match('([\d]+) bathroom\(s\), Living space  ([\d,]+) ft, Lot ([\d,]+)(.*)', listing['Info'])
          if m:
            info['NumberBedrooms'] = ''
            info['NumberBathrooms'] = parseInt(m.groups()[0])
            info['LivingArea'] = parseFloat(m.groups()[1])
            info['LotSize'] = parseFloat(m.groups()[2])
          else:
            m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\), Lot ([\d,]+)(.*)', listing['Info'])
            if m:
              info['NumberBedrooms'] = parseInt(m.groups()[0])
              info['NumberBathrooms'] = parseInt(m.groups()[1])
              info['LivingArea'] = ''
              info['LotSize'] = parseFloat(m.groups()[2])
            else:
              m = re.match('Lot ([\d,]+)(.*)', listing['Info'])
              if m:
                info['NumberBedrooms'] = ''
                info['NumberBathrooms'] = ''
                info['LivingArea'] = ''
                info['LotSize'] = parseFloat(m.groups()[0])
              else:
                m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\)(.*)', listing['Info'])
                if m:
                  info['NumberBedrooms'] = parseInt(m.groups()[0])
                  info['NumberBathrooms'] = parseInt(m.groups()[1])
                  info['LivingArea'] = ''
                  info['LotSize'] = ''
                  print listing
                else:
                  m = re.match('([\d]+) bedroom\(s\), ([\d]+) bathroom\(s\)', listing['Info'])
                  if m:
                    info['NumberBedrooms'] = parseInt(m.groups()[0])
                    info['NumberBathrooms'] = parseInt(m.groups()[1])
                    info['LivingArea'] = ''
                    info['LotSize'] = ''
                  else:
                    print listing
                    pass

  info['Category'] = listing['Category']
  info['AskingPrice'] = parseFloat(listing['AskingPrice'])
  info['PriceSold'] = parseFloat(listing['PriceSold'])

  m = re.match('Sold in (.+) (Days|months|day|month)  on (.*)', listing['SaleDate'])
  if m:
    daysOnMarket = m.groups()[0]
    dateSold = m.groups()[2]
    info['DaysOnMarket'] = daysOnMarket
    info['SaleYYYY'] = parseInt(dateSold.split('-')[0])
    info['SaleMM'] = parseInt(dateSold.split('-')[1])
    info['SaleDD'] = parseInt(dateSold.split('-')[2])
    info['SaleYYYYMMDD'] = info['SaleYYYY']*10000 + info['SaleMM']*100 + info['SaleDD']
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
