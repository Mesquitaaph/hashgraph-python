import numpy as np
from hashlib import md5 as md5_hash
import pickle
from random import randint
from random import seed
import sys
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

sys.setrecursionlimit(5000)

seed(41)
def rand():
  return randint(0, 100000)

N = 2
C = 5

runTime = 2
writeLog = False

"""# Person"""

class Person:
  hashgraph = []
  finishedNodes = []

  def __init__(self, index):
    self.currentRound = 0
    self.index = index

    self.data = Data(index, 0, -1, "\0", "\0", runTime)
    self.tmp = Event(self, self.data)
    self.hashgraph = [self.tmp]

    self.finished_nodes = []
    self.networth = [10000 for i in range(N)]
    self.min_round = 0

    self.hapen_consensus = False

  def getTopNode(self, target):
    "Busca o evento mais novo do alvo, dentre o que self tem guardado"
    top = None
    t = -1
    for i, event in enumerate(self.hashgraph):
      if event.data.owner == target.index and event.data.timestamp > t:
        t = event.data.timestamp
        top = event
    return top

  def createEvent(self, time, gossiper):
    data = Data()

    data.payload = 0
    data.target = -1
    if (rand() % 10) == 0:
      f = rand() / 100
      target = rand() % N
      data.payload = f
      data.target = target

    data.owner = self.index
    data.selfhash = self.getTopNode(self).hash if self.getTopNode(self) else "\0"
    data.gossipHash = self.getTopNode(gossiper).hash if self.getTopNode(gossiper) else "\0"
    data.timestamp = time
    data.position = [data.owner, time]
    tmp = Event(self, data)
    self.hashgraph = [tmp, *self.hashgraph]

  def linkEvents(self, nEvents):
    "Recebe os eventos que o fofocado não conhece + o recém criado e os linka com seus pais"
    for i, nevent in enumerate(nEvents):
      if nevent.selfParent == None and nevent.data.selfhash != "\0":
        c = 0
        for j, event in enumerate(self.hashgraph):
          if event.hash == nevent.data.selfhash:
            nevent.selfParent = event
            c += 1
            if c == 2:
              break
          if event.hash == nevent.data.gossipHash:
            nevent.gossiperParent = event
            c += 1
            if c == 2:
              break

  def outputOrder(self, n, gossiper):
    print(f'\nGossiper: {gossiper.index}')
    print(f'Node owner: {self.hashgraph[n].data.owner} | Timestamp: {self.hashgraph[n].data.timestamp}')
    if self.hashgraph[n].data.payload:
      print(f'Payload: {self.hashgraph[n].data.payload} to {self.hashgraph[n].data.target}')
      networths = ""
      for i in range(N):
        networths += f'{self.networth[i]} '
      print(f'Current Networth: {networths}')
    print(f'\t{self.hashgraph[n].data.selfhash} - Self Parent')
    print(f'\t{self.hashgraph[n].data.gossipHash} - Gossip Parent')
    print(f'\tRound Received: {self.hashgraph[n].roundRecieved}')
    print(f'\tConsensus Time: {self.hashgraph[n].consensusTimestamp}')

  def findUFW(self, witnesses):
    arr = [None for i in range(N)]
    b = [0 for i in range(N)]
    for i, witness in enumerate(witnesses):
      if witness.famous:
        num = witness.data.owner
        if b[num] == 1:
          b[num] = -1
          arr[num] = None
        if b[num] == 0:
          b[num] = 1
          arr[num] = witness
    return arr

  def finalizeOrder(self, n, r, w, gossiper):
    ufw = self.findUFW(w)
    s = []
    j = 0
    while j < N and (not ufw[j] or ufw[j].ancestor(self.hashgraph[n])):
      j += 1
    if j == N:
      for j in range(N):
        if ufw[j]:
          tmp = ufw[j]
          while tmp.selfParent and tmp.selfParent.ancestor(self.hashgraph[n]):
            tmp = tmp.selfParent
          s.append(tmp.data.timestamp)
      if len(s) == 0:
        return True
      
      self.hashgraph[n].roundRecieved = r
      s.sort()
      self.hashgraph[n].consensusTimestamp = np.median(s)
      if self.hashgraph[n].data.payload != 0:
        self.networth[self.hashgraph[n].data.owner] -= self.hashgraph[n].data.payload
        self.networth[self.hashgraph[n].data.target] += self.hashgraph[n].data.payload
      if writeLog:
        self.outputOrder(n, gossiper)

      self.hapen_consensus = True
      return True
    return False

  def removeOldBalls(self):
    size = len(self.hashgraph)
    i = 0
    while i < size:
      # if self.hashgraph[i].consensusTimestamp != -1 and self.hashgraph[i].witness == False:
      #   self.finished_nodes.append(self.hashgraph[i])
      #   self.hashgraph.pop(i)
      #   size -= 1
      #   i -= 1
      if self.hashgraph[i].round < self.currentRound - 3:
        self.min_round = self.hashgraph[i].round+1
        self.hashgraph.pop(i)
        size -= 1
        i -= 1
      i += 1
    # finished_size = len(self.finished_nodes)
    # i = 0
    # while i < finished_size:
    #   if self.finished_nodes[i].round < self.currentRound - 6:
    #     self.finished_nodes.pop(i)
    #     finished_size -= 1
    #     i -= 1
    #   i += 1

  def findOrder(self, gossiper):
    for n in range(len(self.hashgraph)-1, -1, -1):
      if self.hashgraph[n].roundRecieved == -1:
        for r in range(self.hashgraph[n].round, self.hashgraph[0].round+1):
          w = self.findWitnesses(r)
          i = 0
          while i < len(w) and w[i].famous != -1:
            i += 1
          if i == len(w):
            if self.finalizeOrder(n, r, w, gossiper):
              break

  def findWitnesses(self, round):
    witnesses = []

    for i, event in enumerate(self.hashgraph):
      if not (event.round >= round-1):
        break
      if event.round == round and event.witness == True:
        witnesses.append(event)
    return witnesses

  def recieveGossip(self, gossiper, gossip):
    "Recebe a fofoca"
    nEvents:list[Event] = []
    for i in range(len(gossip)):
      n = 0
      e = Event(self, gossip[i])
      while n < len(self.hashgraph):
        if self.hashgraph[n].hash == e.hash:
          break
        n += 1
      if n >= len(self.hashgraph):
        tmp = e
        tmp.data.position = [tmp.data.owner, tmp.data.timestamp]
        self.hashgraph = [tmp, *self.hashgraph]
        nEvents.append(tmp)
    self.createEvent(runTime, gossiper)
    nEvents.append(self.hashgraph[0])
    nEvents.sort(key = lambda e: e.data.timestamp)
    self.linkEvents(nEvents)

    for event in nEvents:
      event.divideRounds()

    self.removeOldBalls()
    for event in nEvents:
      event.decideFame()
    self.findOrder(gossiper)

  def gossip(self, person):
    "Fofoca com outra pessoa"
    arr = []
    self.hashgraph.sort(key = lambda e: e.data.timestamp, reverse=True)
    check = self.getTopNode(person)
    b = [False for i in range(N)]

    # O fofoqueiro verifica os eventos mais novos que ele sabe
    # criado por cada usuario da rede
    for event in self.hashgraph:
      if not b[event.data.owner]:
        if check and check.see(event): # Verifica se o evento do fofocado ve o eventos que eu conheço
          b[event.data.owner] = True;
        arr.append(event.data)

    person.recieveGossip(self, arr)

class Data:
  def __init__(self, owner = 0, payload = 0.0, target = 0, selfhash = "\0", gossipHash = "\0", timestamp = 0):
    self.owner = owner;
    self.payload = payload;
    self.target = target;
    self.selfhash = selfhash;
    self.gossipHash = gossipHash;
    self.timestamp = timestamp;
    self.position = [owner, timestamp]

class Event:
  def __init__(self, person, data):
    self.ancestorsSeen = []
    self.ancestorsNotSeen = []
    self.hashesSeen = []
    self.hashesNotSeen = []
    self.data = data
    self.selfParent = None
    self.gossiperParent = None
    self.consensusTimestamp = -1
    self.roundRecieved = -1
    self.round = 0
    self.witness = True if data.selfhash == "\0" else False
    self.famous = -1
    self.graph = person.hashgraph

    self.hash = self.makeHash()

  def makeHash(self):
    return md5_hash(pickle.dumps(self.data)).hexdigest()

  def ancestor(self, y):
    done = False
    yHash = self.hash
    visited = []
    if yHash in self.hashesSeen or yHash in self.ancestorsSeen:
      return True
    if yHash in self.ancestorsNotSeen:
      return False
    b = self.ancestorRecursion(y, done, visited)
    if not b:
      self.ancestorsNotSeen = [yHash, *self.ancestorsNotSeen]
    else:
      self.ancestorsSeen = [yHash, *self.ancestorsSeen]
    return b

  def ancestorRecursion(self, y, done, visited):
    if done:
      return True
    if self.hash == y.hash:
      done = True
      return True
    if self.hash in visited:
      return False
    visited.append(self.hash)
    if self.data.timestamp < y.data.timestamp:
      return False
    if not self.selfParent or not self.gossiperParent:
      return False
    return (self.selfParent.ancestorRecursion(y, done, visited) or
            self.gossiperParent.ancestorRecursion(y, done, visited))

  def stronglySee(self, y):
    numSee = 0
    found = [False for i in range(N)]
    for n in range(len(self.graph)):
      if found[self.graph[n].data.owner]: #or self.graph[n].round < y.round:
        continue
      if self.see(self.graph[n]) and self.graph[n].see(y):
        numSee += 1
        found[self.graph[n].data.owner] = True
        if numSee > 2*N/3:
          return True
    return False

  def decideFame(self):
    if not self.witness or self.round < 2:
      return
    for x in range(len(self.graph)-1, -1, -1):
      if self.graph[x].witness and self.graph[x].famous == -1 and self.round > self.graph[x].round:
        s = people[self.data.owner].findWitnesses(self.round-1)
        count = 0
        countNo = 0
        for y in range(len(s)):
          if self.stronglySee(s[y]):
            if s[y].see(self.graph[x]):
              count += 1
            else:
              countNo += 1
        d = self.round - self.graph[x].round
        if count > 2*N/3:
          self.graph[x].famous = 1
        elif countNo > 2*N/3:
          self.graph[x].famous = 0
        elif d % C == 0:
          self.graph[x].famous = int(self.graph[x].hash[16], 16) % 2

  def divideRounds(self):
    "Atualiza o round do evento fofocado"

    if not self.selfParent or not self.gossiperParent:
      self.round = 0
      return

    self.round = self.selfParent.round
    if self.gossiperParent.round > self.round:
      self.round = self.gossiperParent.round
    numStrongSee = 0
    witnesses = people[self.data.owner].findWitnesses(self.round)

    for i, witness in enumerate(witnesses):
      if numStrongSee > 2 * N / 3:
        break
      if self.stronglySee(witness):
        numStrongSee += 1
    if numStrongSee > 2*N/3:
      self.round += 1
      if people[self.data.owner].currentRound < self.round:
        people[self.data.owner].currentRound += 1

    # Define se evento é testemunha: se for o primeiro na historia ou se for o primeiro do round
    self.witness = self.selfParent == None or self.selfParent.round < self.round

  def seeRecursion(self, y, forkCheck, done, visited):
    "Retorna True se o evento é ancestral dele mesmo"
    if self.hash in visited:
      return False
    visited.append(self.hash)
    if self.data.owner == y.data.owner:
      forkCheck.append(self)
    if done:
      return True
    if self.hash == y.hash:
      done = True
      return True
    if self.data.timestamp < y.data.timestamp:
      return False
    if not self.selfParent:
      return False
    if not self.gossiperParent:
      return False
    return (self.selfParent.seeRecursion(y, forkCheck, done, visited) or
            self.gossiperParent.seeRecursion(y, forkCheck, done, visited))

  def see(self, y):
    yHash = y.hash
    if yHash in self.hashesSeen:
      return True
    if yHash in self.hashesNotSeen:
      return False

    forkCheck = []
    visited = []
    done = False
    b = self.seeRecursion(y, forkCheck, done, visited)

    if b == False:
      self.hashesNotSeen.append(yHash)
      return False    
    
    self.hashesSeen.append(yHash)

    return True

def plot_graph(person, transition_time):
  plt.clf()
  plt.xticks([0, 1, 2, 3, 4, 5])
  plt.yticks([0])
  plt.grid()



  G = nx.Graph()
  edges = []
  positions = {}
  labels = {}
  for _, event in enumerate(person.hashgraph):
    positions[event.hash] = event.data.position
    labels[event.hash] = event.round
    if event.data.gossipHash != '\0' and event.round - 1 >= person.min_round:
      edges.append((event.hash, event.data.gossipHash))

  # if len(person.hashgraph) > 100:
  #   hashes = [event.hash for event in person.hashgraph]
  #   hashes.sort()
  #   print(hashes)
  # print(len(person.hashgraph),len(positions.keys()))
  # for _, event in enumerate(person.hashgraph):
  #   if event.data.gossipHash in positions.keys():
  #     edges.append((event.hash, event.data.gossipHash))

  G.add_edges_from(edges)
  options = {"node_size": 100, 
              "node_color": "black", 
              "labels": labels,
              "font_size": 8,
              "font_color": "white"}
  nx.draw_networkx(G, positions, cmap = plt.get_cmap('jet'), with_labels=True, **options)
  plt.pause(transition_time)
  # plt.show()
  # figure.canvas.draw()

people = [Person(i) for i in range(N)]

personShown = 0
gossipsCounted = 0

elapsed = 0
sequences = []
consensus_events = 0
n_consensus = []

show_graph = True

print(f'N = {N}')
while True:
  i = rand() % N
  j = rand() % N
  while j == i:
    j = rand() % N

  start = datetime.now().timestamp()

  people[i].gossip(people[j])
  gossipsCounted += 1
  consensus_events += 1

  elapsed += datetime.now().timestamp() - start

  if gossipsCounted >= 100 and len(sequences) < 20:
    gossipPerSec = gossipsCounted / elapsed
    gossipsCounted = 0
    # print(f'{gossipPerSec} fofoca por segundo')

    elapsed = 0
    sequences.append(gossipPerSec)
    if len(sequences) == 20:
      print('sequences', sequences)

  if people[j].hapen_consensus and len(n_consensus) < 50:
    # print(f'{consensus_events} eventos ate consenso')
    people[j].hapen_consensus = False
    n_consensus.append(consensus_events)
    consensus_events = 0
    if len(n_consensus) == 50:
      print('n_consensus', n_consensus)

  if len(n_consensus) == 50 and len(sequences) == 20:
    break

  if show_graph:
    start_plot = datetime.now().timestamp()
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.yticks([0])
    plt.grid()
    plot_graph(people[0], 0.1)
    elapsed_plot = datetime.now().timestamp() - start_plot
  
  runTime += 1
