from collections import defaultdict

class Beurs:
	""" deze klasse representeert een beurs. En diens beschikbare markten """
	#wens: toevoegen van beurs is slechts het toevoegen van een lijst markten
	def __init__(self,name,markets,base_position):
		self.name = name  # als je dan zegt T = Beurs("Bittrex"), dan kun je vervolgens T.name roepen om er achter te komen dat T Bittrex is
		self.base = base_position #geeft aan of de base vooraan (0) of achteraan(1) de markt-string staat
		self.markets = sorted([set_base(market,self.base) for market in markets])
		# market values
	def remove_market(self,market):
		#met deze functie haal je markten uit de markt-lijst die alleen in de marktlijst van deze beurs voorkomen
		if market in self.markets:
			self.markets.remove(market)
		
	#def add_market(self,market):
	#	# met deze functie kun je een enkele markt toevoegen aan de beurs. Als je dat zou willen
	#	self.markets.append(market)
			

#class Markt:
#	""" deze klasse geeft de markt van twee munten weer"""
#	# niet alle beurzen hebben base op de eerste plek staan. schrijf een functie waarmee je makkelijk aan kan geven of base op 0 of 1 staat
#	def __init__(self, base_asset, quote_asset):
#		self.base = base_asset # brood
#		self.quote = quote_asset # geld
		
def set_base(market,base_position):
	""" met deze functie kun je eenvoudig de marktparen juist sorteren """
	#market = eg EEE-BBB
	base = market.split("-")[base_position]
	quote = market.split("-")[(base_position-1)**2]
	return "-".join([base,quote]) # of bijvoorbeeld return (base,quote), als tuples makkelijker werken
	
# van alle koppels de buy en sell waarden ophalen, en met elkaar vergelijken

#load data
data = ["BIT-UTC", "BIT-PIT", "BIT-QEM", "ETC-KLM", "ETC-QEM"]
data2 = ["PIT-ETC", "ETC-BIT", "QEM-ETC", "KLM-ETC", "QEM-BIT"]
data3 = ["PIT-BIT", "ETC-BIT", "QEM-BIT", "QEM-KLM", "UTC-ETC"]
# define exchanges
exchanges = [Beurs("Bittrex",data,0), Beurs("Binance",data2,1), Beurs("Poloniex",data3,1)]

# determine 'singles'
market_dict = defaultdict(list)
for exchange in exchanges:
	for market in exchange.markets:
		market_dict[market].append(1)

# remove 'singles',remember the rest
all_markets = []
for exchange in exchanges:
	for market in market_dict:
		if len(market_dict[market])<=1:
			exchange.remove_market(market)
		else:
			all_markets.append(market)

# show shared markets between exchanges
for exchange in exchanges:
	print exchange.name,exchange.markets

print all_markets