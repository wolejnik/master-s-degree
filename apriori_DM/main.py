from csv import reader
from collections import defaultdict
from itertools import chain, combinations


# Wczytanie danych z pliku CSV
def getFromFile(fname):
    itemSets = []
    itemSet = set()

    with open(fname, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            record = set(line)
            for item in record:
                itemSet.add(frozenset([item]))
            itemSets.append(record)
    return itemSet, itemSets


# Wybieramy produktów, których wsparcie jest większe od zadanego
def getAboveMinSup(itemSet, itemSetList, minSup, globalItemSetWithSup):
    freqItemSet = set()
    localItemSetWithSup = defaultdict(int)

    for item in itemSet:
        for itemSet in itemSetList:
            if item.issubset(itemSet):
                globalItemSetWithSup[item] += 1
                localItemSetWithSup[item] += 1
    for item, supCount in localItemSetWithSup.items():
        support = float(supCount / len(itemSetList))
        if(support >= minSup):
            freqItemSet.add(item)
            print(f'{item} ==> {support}')

    return freqItemSet


# Przechodzimy przez wszystkie możliwe podzbiory w zestawie elementów.
# Jeśli podzbioru nie ma w poprzednim zestawie częstych elementów, przycinamy go.
def pruning(candidateSet, prevFreqSet, length):
    tempCandidateSet = candidateSet.copy()
    for item in candidateSet:
        subsets = combinations(item, length)
        for subset in subsets:
            if(frozenset(subset) not in prevFreqSet):
                tempCandidateSet.remove(item);
                break

    return tempCandidateSet


def associationRule(freqItemSet, itemSetWithSup, minConf):
    rules = []
    for k, itemSet in freqItemSet.items():
        for item in itemSet:
            subsets = chain.from_iterable(combinations(item, r) for r in range(1, len(item)))
            for s in subsets:
                confidence = float(
                    itemSetWithSup[item] / itemSetWithSup[frozenset(s)])
                if(confidence > minConf):
                    rules.append([set(s), set(item.difference(s)), confidence])
    return rules



def apriori(fname, minSup, minConf):
    itemSet, itemSetList = getFromFile(fname)

    globalFreqItemSet = dict() # czestotliwość produktu
    globalItemSetWithSup = defaultdict(int) # czestotliwość zestawów z wsparciem

    itemCurrentSetTmp = getAboveMinSup(
        itemSet, itemSetList, minSup, globalItemSetWithSup)
    currentLSet = itemCurrentSetTmp
    k = 2

    # Calculating frequent item set
    while(currentLSet):
        # Storing frequent itemset
        globalFreqItemSet[k-1] = currentLSet
        # candidateSet = getUnion(currentLSet, k)
        candidateSet = set()
        # candidateSet = set([i.union(j)
        for i in currentLSet:
            for j in currentLSet:
                if len(i.union(j)) == k:
                    candidateSet.add(i.union(j))

        # Perform subset testing and remove pruned supersets
        candidateSet = pruning(candidateSet, currentLSet, k-1)
        # Scanning itemSet for counting support
        currentLSet = getAboveMinSup(
            candidateSet, itemSetList, minSup, globalItemSetWithSup)
        k += 1

    rules = associationRule(globalFreqItemSet, globalItemSetWithSup, minConf)
    rules.sort(key=lambda x: x[2])

    return globalFreqItemSet, rules

if __name__ == "__main__":

    freqItemSet, rules = apriori('store_data.csv', 0.05, 0.1)

    # print(rules)
    print('\n------------------------------------\n')
    for rule in rules:
        print(f'If {rule[0] } then {rule[1]} ==> {rule[2]}')