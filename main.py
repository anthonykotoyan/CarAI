import math
import random
import time

from nn import NeuralNetwork as nn
import pygame

pygame.init()
width, length = 1300, 800
screen = pygame.display.set_mode((width, length))
clock = pygame.time.Clock()
running = True
waitTime = .5
ct = 0
dt = clock.tick(60) / 1000
trainMode = False
randomStart = True

carImage = pygame.image.load('car.png').convert()
savedNN = [[8, 4, 3, 5], [
    [0.29864639568192586, -0.1434398543558178, 0.9945055512476999, -0.6752886755290843, 0.8586208116279508,
     -0.57637091750516, 0.2571771204436239, -0.835329332828993, 0.7188273717231288, -0.3678678679666271,
     0.7857774371677357, -0.9405637449677452, -0.09323130302943944, -0.7460546220530254, -0.0892025742824969,
     -0.1822366428455992, -0.8552939136866897, -0.5527503142362009, -0.28113401214845823, 0.9227478717902329,
     -0.9857701136828134, -0.23318570597685018, 0.4091380611439134, 0.13450139875214584, -1.0, 0.7954905597115534,
     0.18992646421280432, 0.90469783307501, 0.1060858919398096, -0.3589228914948741, -0.8591588648886772,
     0.2196958913944953],
    [0.4677640014681874, -0.27433039103116647, -0.9358874289508918, -0.37207461773273687, -0.39594942143648515,
     0.5682346282774422, -0.6407738936087923, 0.34463350104138657, 0.23640754403161518, 0.7700615657623091,
     0.4884873817190012, -1.0],
    [0.6696318324327646, -0.00023538767766331792, 1.0, -0.7696903212123514, -0.4635985405716181, -0.030250836944431425,
     -0.6352742727862566, 0.6647275556795049, -0.9306975016746314, 0.016067446720762824, -0.7429361486813628,
     -0.23691153868230963, -0.6323014601274262, -0.16009692883785198, -0.4168847415136353]],
           [[0.2908870558593204, -0.5282570287948274, 0.9676530357466885, -0.8592323226569096],
            [-0.1770539682463072, -0.4158209269052165, 0.6524301493994362],
            [1.0, -0.8494588135097656, 0.2517490520434555, -0.310137110308507, -0.1996874989726426]]]

track = [[(98, 146), (112, 140), (170, 131), (240, 114), (308, 100), (362, 87), (420, 88), (481, 122), (513, 163),
          (571, 190), (650, 209), (701, 215), (753, 218), (814, 201), (899, 170), (979, 134), (1123, 121), (1166, 116),
          (1211, 140), (1246, 176), (1260, 239), (1261, 357), (1254, 447), (1262, 564), (1281, 655), (1271, 687),
          (1233, 745), (1173, 780), (1131, 786), (1061, 785), (994, 779), (916, 759), (842, 715), (788, 682),
          (736, 659), (674, 639), (593, 647), (522, 682), (473, 701), (417, 718), (347, 740), (259, 745), (160, 717),
          (80, 601), (72, 582), (50, 510), (51, 447), (51, 383), (50, 326), (50, 256), (72, 155), (98, 146)],
         [(247, 648), (294, 656), (361, 656), (437, 635), (491, 605), (527, 583), (613, 550), (673, 543), (777, 556),
          (847, 597), (906, 639), (1015, 688), (1136, 675), (1171, 630), (1176, 565), (1159, 506), (1158, 443),
          (1152, 328), (1151, 251), (1105, 200), (1044, 190), (974, 225), (899, 267), (878, 273), (833, 292),
          (783, 313), (716, 331), (661, 337), (610, 311), (547, 286), (484, 267), (459, 249), (446, 218), (416, 195),
          (381, 179), (309, 158), (276, 168), (232, 192), (185, 206), (160, 241), (160, 287), (148, 328), (130, 372),
          (117, 427), (128, 495), (154, 539), (181, 571), (225, 616), (247, 648)], (729, 611),
         [[(761, 546), (743, 712)], [(776, 542), (771, 702)], [(822, 548), (813, 725)], [(897, 607), (836, 766)],
          [(938, 635), (877, 762)], [(1000, 657), (898, 776)], [(1023, 672), (949, 785)], [(1037, 672), (1007, 799)],
          [(1079, 659), (1049, 799)], [(1102, 669), (1102, 794)], [(1131, 648), (1120, 792)],
          [(1155, 663), (1184, 786)], [(1135, 627), (1260, 743)], [(1144, 616), (1290, 666)],
          [(1152, 584), (1278, 619)], [(1163, 570), (1274, 565)], [(1161, 550), (1263, 540)],
          [(1180, 526), (1290, 492)], [(1158, 485), (1295, 460)], [(1130, 420), (1294, 414)],
          [(1120, 367), (1261, 379)], [(1133, 305), (1277, 288)], [(1142, 271), (1274, 245)],
          [(1125, 245), (1267, 128)], [(1115, 230), (1212, 132)], [(1102, 222), (1168, 114)],
          [(1078, 213), (1110, 109)], [(1075, 216), (1060, 115)], [(1037, 241), (1030, 113)], [(1012, 240), (988, 131)],
          [(975, 245), (934, 163)], [(962, 265), (858, 151)], [(908, 328), (803, 154)], [(817, 327), (763, 184)],
          [(777, 358), (733, 170)], [(708, 372), (698, 202)], [(661, 353), (681, 176)], [(613, 357), (669, 177)],
          [(548, 329), (632, 160)], [(493, 302), (625, 119)], [(452, 255), (604, 128)], [(436, 235), (577, 139)],
          [(405, 223), (528, 140)], [(419, 216), (500, 117)], [(402, 209), (466, 101)], [(380, 207), (463, 94)],
          [(357, 195), (424, 97)], [(347, 201), (357, 64)], [(323, 184), (326, 84)], [(305, 189), (284, 101)],
          [(283, 202), (258, 112)], [(258, 222), (206, 88)], [(226, 199), (181, 101)], [(208, 255), (145, 108)],
          [(208, 242), (90, 118)], [(182, 228), (60, 170)], [(181, 261), (41, 205)], [(188, 280), (42, 243)],
          [(149, 302), (24, 269)], [(153, 339), (46, 322)], [(169, 388), (36, 374)], [(133, 406), (48, 410)],
          [(135, 439), (31, 456)], [(176, 463), (44, 494)], [(143, 489), (49, 555)], [(149, 521), (20, 591)],
          [(192, 520), (88, 632)], [(220, 553), (115, 692)], [(235, 593), (165, 721)], [(308, 598), (231, 761)],
          [(345, 654), (345, 753)], [(367, 609), (449, 778)], [(480, 579), (507, 741)], [(534, 552), (565, 704)],
          [(583, 492), (620, 717)], [(647, 519), (599, 735)], [(690, 534), (689, 672)]]]


def sign(x):
    if x != 0:
        return abs(x) / x
    return 0


def line_intersection(a, b, c, d):
    d1 = (
            (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))
    d2 = (
            (a[0] - b[0]) * (c[1] - d[1]) - (a[1] - b[1]) * (c[0] - d[0]))
    if d1 == 0 or d2 == 0:
        return [False, []]
    else:
        t = ((a[0] - c[0]) * (c[1] - d[1]) - (a[1] - c[1]) * (c[0] - d[0])) / d1
        u = ((a[0] - c[0]) * (a[1] - b[1]) - (a[1] - c[1]) * (a[0] - b[0])) / d2

    # check if line actually intersect
    if 0 <= t <= 1 and 0 <= u <= 1:
        return [True, (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))]
    else:
        return [False, []]


def BoxCollision(corners, line):
    for i in range(4):
        if i == 3:
            j = -1
        else:
            j = i
        boxLine = [corners[j], corners[j + 1]]
        collisionInfo = line_intersection(line[0], line[1], boxLine[0], boxLine[1])
        if collisionInfo[0]:
            return True

    return False


def DrawCheckPoints(cp):
    for i in range(len(cp)):
        if len(cp[i]) == 2:
            pygame.draw.line(screen, "green", cp[i][0], cp[i][1], 1)


def DrawTrack(track, drawCP):
    for wall in range(2):
        for i in range(len(track[wall]) - 1):
            pygame.draw.line(screen, "black", track[wall][i], track[wall][i + 1], 2)
    pygame.draw.circle(screen, 'green', track[2], 3)
    if drawCP:
        DrawCheckPoints(track[3])


layers = [len([-90, -37.5, -15, 0, 15, 37.5, 90]) + 1, 4, 3, 5]

acceleration = 3
breakStrength = 1
maxSpeed = 9
driftLength = 10
driftLength = 4
turnSpeed = 4
driftFriction = 14
friction = 0.4
baseTSValue = -.3
ShiftUpTS = 1.1
driftSpeed = .6
minDriftAngle = 20
normalSlip = .3
handBrakeSlip = normalSlip * 1.5
errorCorrectionStrength = 15


class Agent:
    gen = 0
    genLength = 30
    startTime = ct
    boxSize = 2

    numBestAgents = 3
    mutateFactor = .34
    mutationChance = .3
    allAgents = []
    allMaxCP = 0
    nnBest = 0

    bestFitness = 0

    startAngle = 10

    def __init__(self):
        self.image = pygame.image.load('babyman.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 15, self.size * 15))
        self.acc = acceleration
        self.turnSpeed = turnSpeed
        self.slip = normalSlip
        sa = random.uniform(-Agent.startAngle, Agent.startAngle)
        self.dir = sa
        self.angle = sa
        self.vel = pygame.Vector2(0, 0)
        self.pos = pygame.Vector2(track[2][0], track[2][1])
        self.speed = 0
        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        self.hbColor = "green"
        self.carCorners = [(self.pos.x - self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize),
                           (self.pos.x - self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize)]
        self.angles = [-90, -37.5, -15, 0, 15, 37.5, 90]
        self.vision = []
        self.fitness = 0
        self.nextCP = 0
        self.runAgent = True

        if randomStart:
            self.NN = nn(layers)
            self.NN.randomize()

        else:
            self.NN = copyNN(savedNN)
            self.NN.values(self.NN.mutate(Agent.mutationChance, Agent.mutateFactor))

        Agent.allAgents.append(self)

    def Vision(self):

        positions = []
        endPositions = []
        rayDist = 100000
        dists = []
        for i in range(len(self.angles)):
            pos = pygame.Vector2(math.cos(math.radians(self.angle + self.angles[i])) * rayDist + self.pos.x,
                                 math.sin(math.radians(self.angle + self.angles[i])) * rayDist + self.pos.y)
            positions.append(pos)
            endPos = pos
            wallDists = []
            for walls in range(2):
                for wall in range(len(track[walls]) - 1):
                    intersection = line_intersection(self.pos, pos, track[walls][wall], track[walls][wall + 1])
                    if intersection[0]:
                        wallDist = math.dist(self.pos, pygame.Vector2(intersection[1][0], intersection[1][1]))
                        wallDists.append(wallDist)

                        if wallDist == min(wallDists):
                            endPos = pygame.Vector2(intersection[1][0], intersection[1][1])
            if wallDists:
                dists.append(min(wallDists))
            else:
                dists.append(0)
            endPositions.append(endPos)
        return [dists, endPositions]

    def DrawVision(self, endPositions):
        for i in range(len(endPositions)):
            pygame.draw.line(screen, "purple", self.pos, endPositions[i], 1)
            pygame.draw.circle(screen, "orange", endPositions[i], 5)

    def HitBox(self, draw):
        if draw:
            for i in range(4):
                if i == 3:
                    j = -1
                else:
                    j = i

                pygame.draw.line(screen, self.hbColor, self.carCorners[j], self.carCorners[j + 1], 2)

    def DrawCar(self):

        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        rotated_image = pygame.transform.rotate(self.image, -(self.angle + 90))
        # Update the car's rectangle to the new rotated image's rectangle
        self.car = rotated_image.get_rect(center=self.car.center)
        # Draw the rotated image onto the screen
        screen.blit(rotated_image, self.car)

    def DriftTrail(self, dist, wid):
        driftAngle = abs(((self.angle - self.dir + 540) % 360 - 180))
        if driftAngle >= minDriftAngle and abs(self.speed) > driftSpeed:
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x + self.size,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (40, 40, 30), self.pos, endPos, wid)
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x - self.size,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (40, 40, 30), self.pos, endPos, wid)

    def DrawAngle(self, dist, draw):
        if draw:
            endPos = pygame.Vector2(math.cos(math.radians(self.dir)) * dist + self.pos.x,
                                    math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, "blue", self.pos, endPos, self.size)
            endPos = pygame.Vector2(math.cos(math.radians(self.angle)) * dist + self.pos.x,
                                    math.sin(math.radians(self.angle)) * dist + self.pos.y)
            pygame.draw.line(screen, "red", self.pos, endPos, self.size)

    def ApplyVelocity(self):
        self.carCorners = [(self.pos.x - self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y - self.size * Agent.boxSize),
                           (self.pos.x + self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize),
                           (self.pos.x - self.size * Agent.boxSize, self.pos.y + self.size * Agent.boxSize)]
        self.vision = self.Vision()

        self.pos.x += self.vel.x
        self.pos.y += self.vel.y

    def ResetAgent(self):
        sa = random.uniform(-Agent.startAngle, Agent.startAngle)
        self.dir = sa
        self.angle = sa
        self.speed = 0
        self.fitness = 0
        self.nextCP = 0
        self.runAgent = True
        self.pos = pygame.Vector2(track[2][0], track[2][1])

    def AgentDeath(self):
        if trainMode:
            self.runAgent = False
        else:
            self.ResetAgent()

    def TrackCollisions(self):
        for walls in range(2):
            for wall in range(len(track[walls]) - 1):

                coll = BoxCollision(self.carCorners, (track[walls][wall], track[walls][wall + 1]))
                if coll:
                    self.fitness -= 1
                    self.AgentDeath()

    def TrackCheckpoints(self, draw):
        coll = BoxCollision(self.carCorners, track[3][self.nextCP])
        if draw:
            pygame.draw.line(screen, "green", track[3][self.nextCP][0], track[3][self.nextCP][1], 1)
        if coll:
            self.fitness += 1
            if self.nextCP == len(track[3]) - 1:
                self.nextCP = 0
            else:
                self.nextCP += 1

    def BorderCollisions(self):
        if 0 >= self.pos.x:
            self.AgentDeath()

        if self.pos.x >= width:
            self.AgentDeath()

        if 0 >= self.pos.y:
            self.AgentDeath()

        if self.pos.y >= length:
            self.AgentDeath()

    def ApplyDirection(self):
        if abs(self.speed) >= maxSpeed:
            self.speed = maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, outputs):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if outputs[0]:
            self.speed -= self.acc * dt
        if abs(self.speed) > driftSpeed:
            self.speed += driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += friction * -sign(self.speed) * dt

        if outputs[4]:
            self.slip = handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt * breakStrength
            else:
                self.speed = 0
        else:
            self.slip = normalSlip

        if outputs[1]:
            if self.speed < 0:
                self.speed += self.acc * dt * breakStrength
            else:
                self.speed = 0
        if outputs[2]:
            # angle error changes positively
            self.angle += self.turnSpeed
            if self.speed != 0:
                self.dir += self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError < 0:
                self.dir += sign(angleError) * errorCorrectionStrength * (abs(angleError) / 180)


        elif outputs[3]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (driftLength * .25)
        # b = 4 * axTurnSpeed / maxSpeed ** 2
        # self.turnSpeed = -b * (baseTSValue + abs(self.speed)) * (
        #         baseTSValue - maxSpeed + abs(self.speed)) + ShiftUpTS

    @staticmethod
    def RestartGen():
        Agent.gen += 1
        print([layers, Agent.nnBest[0], Agent.nnBest[1]])
        fitnessMin = 1
        Agent.startTime = ct
        # a list of all agents fitness
        allFitness = [agent.fitness for agent in Agent.allAgents]

        # loop through all agents to update nn

        for agent in Agent.allAgents:

            if max(allFitness) > fitnessMin:

                if Agent.nnBest[0] != agent.NN.weights or Agent.nnBest[1] != agent.NN.biases:
                    agent.NN.weights = Agent.nnBest[0]
                    agent.NN.biases = Agent.nnBest[1]
                    agent.NN.values(agent.NN.mutate(Agent.mutationChance, Agent.mutateFactor))

                agent.ResetAgent()
            else:

                agent.NN = nn(layers)
                agent.NN.randomize()
                agent.ResetAgent()
        print(Agent.gen)

    def CheckIfBad(self):
        if ct - Agent.startTime > Agent.genLength * .25:
            if self.fitness <= 0:
                self.AgentDeath()
        for i in range(3):
            if ct - Agent.startTime > Agent.genLength * (.25 * (i + 1)):
                if abs(self.speed) <= .5:
                    self.AgentDeath()

    @staticmethod
    def ManageGen():
        if ct - Agent.startTime > Agent.genLength:
            Agent.RestartGen()

            Agent.startTime = ct

    def RunAgent(self):
        nnInputs = self.vision[0]
        nnInputs.append(self.speed)
        nnOutputs = self.NN.run(nnInputs, nn.Tanh)

        agentInput = []
        for i in range(len(nnOutputs)):
            if nnOutputs[i] > 0:
                agentInput.append(True)
            else:
                agentInput.append(False)

        self.Controls(agentInput)

    @staticmethod
    def UpdateAgents():

        Agent.ManageGen()
        deadAgents = 0
        if pygame.key.get_pressed()[pygame.K_c]:
            we_ = Agent.nnBest[0]
            ba_ = Agent.nnBest[1]
            print([layers, we_, ba_])
        for agent in Agent.allAgents:
            if agent.runAgent:

                agent.ApplyVelocity()
                agent.ApplyDirection()
                agent.RunAgent()

                agent.CheckIfBad()

                agent.TrackCheckpoints(True)
                agent.TrackCollisions()
                agent.BorderCollisions()

                agent.DriftTrail(10 * agent.size, 5 * agent.size)
                agent.DrawAngle(50, False)
                agent.DrawCar()

            else:

                deadAgents += 1
        if deadAgents >= len(Agent.allAgents):
            Agent.RestartGen()

        allFitness = [agent.fitness for agent in Agent.allAgents]

        Agent.nnBest = [list(Agent.allAgents[allFitness.index(max(allFitness))].NN.weights),
                        list(Agent.allAgents[allFitness.index(max(allFitness))].NN.biases)]

        Agent.allAgents[allFitness.index(max(allFitness))].HitBox(True)
        Agent.allAgents[allFitness.index(max(allFitness))].DrawVision(
            Agent.allAgents[allFitness.index(max(allFitness))].vision[1])

    def TrainedMode(self):
        self.ApplyVelocity()
        self.ApplyDirection()
        self.RunAgent()
        # self.DrawVision(self.vision[1])
        self.TrackCheckpoints(False)
        self.TrackCollisions()
        self.BorderCollisions()

        self.DriftTrail(10 * self.size, 5 * self.size)
        self.DrawAngle(50, False)
        self.DrawCar()


def copyNN(network):
    copiedNetwork = nn(network[0])
    copiedNetwork.weights = network[1]
    copiedNetwork.biases = network[2]
    return copiedNetwork


numAgents = 25
ag = 0

if trainMode:
    for i in range(numAgents):
        Agent()
else:
    ag = Agent()
    ag.NN = copyNN(savedNN)  # put a good nn


class Car:
    boxSize = 2.5

    def __init__(self):
        self.image = pygame.image.load('car.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 10, self.size * 10))
        self.acc = acceleration
        self.turnSpeed = turnSpeed
        self.slip = normalSlip
        self.dir = 0
        self.angle = 0
        self.vel = pygame.Vector2(0, 0)
        self.pos = pygame.Vector2(track[2][0], track[2][1])
        self.speed = 0
        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        self.hbColor = "green"
        self.carCorners = []
        self.vision = []

    def Vision(self):
        angles = [-80, -45, -15, -5, 0, 5, 15, 45, 80]
        positions = []
        endPositions = []
        rayDist = 100000
        dists = []
        for i in range(len(angles)):
            pos = pygame.Vector2(math.cos(math.radians(self.angle + angles[i])) * rayDist + self.pos.x,
                                 math.sin(math.radians(self.angle + angles[i])) * rayDist + self.pos.y)
            positions.append(pos)
            endPos = pos
            wallDists = []
            for walls in range(2):
                for wall in range(len(track[walls]) - 1):
                    intersection = line_intersection(self.pos, pos, track[walls][wall], track[walls][wall + 1])
                    if intersection[0]:
                        wallDist = math.dist(self.pos, pygame.Vector2(intersection[1][0], intersection[1][1]))
                        wallDists.append(wallDist)

                        if wallDist == min(wallDists):
                            endPos = pygame.Vector2(intersection[1][0], intersection[1][1])
            dists.append(min(wallDists))
            endPositions.append(endPos)
        return [dists, endPositions]

    def DrawVision(self, endPositions):
        for i in range(len(endPositions)):
            pygame.draw.line(screen, "purple", self.pos, endPositions[i], 1)
            pygame.draw.circle(screen, "orange", endPositions[i], 5)

    def HitBox(self, draw):
        if draw:
            for i in range(4):
                if i == 3:
                    j = -1
                else:
                    j = i

                pygame.draw.line(screen, self.hbColor, self.carCorners[j], self.carCorners[j + 1], 2)

    def DrawCar(self):

        self.car = self.image.get_rect(center=(self.pos.x, self.pos.y))
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        # Update the car's rectangle to the new rotated image's rectangle
        self.car = rotated_image.get_rect(center=self.car.center)
        # Draw the rotated image onto the screen
        screen.blit(rotated_image, self.car)

    def DriftTrail(self, dist, wid):
        driftAngle = abs(((self.angle - self.dir + 540) % 360 - 180))
        if driftAngle >= minDriftAngle and abs(self.speed) > driftSpeed:
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x + self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)
            endPos = pygame.Vector2(
                sign(self.speed) * math.cos(math.radians(self.dir)) * dist + self.pos.x - self.size * 2,
                sign(self.speed) * math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, (30, 30, 30), self.pos, endPos, wid)

    def DrawAngle(self, dist, draw):
        if draw:
            endPos = pygame.Vector2(math.cos(math.radians(self.dir)) * dist + self.pos.x,
                                    math.sin(math.radians(self.dir)) * dist + self.pos.y)
            pygame.draw.line(screen, "blue", self.pos, endPos, self.size)
            endPos = pygame.Vector2(math.cos(math.radians(self.angle)) * dist + self.pos.x,
                                    math.sin(math.radians(self.angle)) * dist + self.pos.y)
            pygame.draw.line(screen, "red", self.pos, endPos, self.size)

    def ApplyVelocity(self):
        self.carCorners = [(self.pos.x - self.size * Car.boxSize, self.pos.y - self.size * Car.boxSize),
                           (self.pos.x + self.size * Car.boxSize, self.pos.y - self.size * Car.boxSize),
                           (self.pos.x + self.size * Car.boxSize, self.pos.y + self.size * Car.boxSize),
                           (self.pos.x - self.size * Car.boxSize, self.pos.y + self.size * Car.boxSize)]
        self.vision = self.Vision()

        self.pos.x += self.vel.x
        self.pos.y += self.vel.y

    def ResetPos(self):
        self.angle = 0
        self.dir = 0
        self.speed = 0
        self.pos = pygame.Vector2(track[2][0], track[2][1])

    def TrackCollisions(self):
        for walls in range(2):
            for wall in range(len(track[walls]) - 1):
                coll = BoxCollision(self.carCorners, (track[walls][wall], track[walls][wall + 1]))
                if coll:
                    self.ResetPos()

    def BorderCollisions(self):
        if 0 >= self.pos.x:
            self.ResetPos()

        if self.pos.x >= width:
            self.ResetPos()

        if 0 >= self.pos.y:
            self.ResetPos()

        if self.pos.y >= length:
            self.ResetPos()

    def ApplyDirection(self):
        if abs(self.speed) >= maxSpeed:
            self.speed = maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, keys):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if keys[pygame.K_w]:
            self.speed -= self.acc * dt
        if abs(self.speed) > driftSpeed:
            self.speed += driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += friction * -sign(self.speed) * dt
        if keys[pygame.K_SPACE]:
            self.slip = handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt * breakStrength
            else:
                self.speed = 0
        else:
            self.slip = normalSlip

        if keys[pygame.K_s]:
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        if keys[pygame.K_d]:
            # angle error changes positively
            self.angle += self.turnSpeed
            if self.speed != 0:
                self.dir += self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError < 0:
                self.dir += sign(angleError) * errorCorrectionStrength * (abs(angleError) / 180)


        elif keys[pygame.K_a]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (driftLength * .25)
        # b = 4 * Car.maxTurnSpeed / Car.maxSpeed ** 2
        # self.turnSpeed = -b * (Car.baseTSValue + abs(self.speed)) * (
        #         Car.baseTSValue - Car.maxSpeed + abs(self.speed)) + Car.ShiftUpTS

    def UpdateCar(self, keys):

        self.ApplyVelocity()
        self.ApplyDirection()
        self.Controls(keys)
        self.TrackCollisions()
        self.BorderCollisions()

        # self.DrawVision(self.vision[1])
        self.DriftTrail(10 * self.size, 4 * self.size)
        self.DrawAngle(50, True)
        self.DrawCar()
        # self.HitBox(True)


car1 = Car()
wait = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    if wait:
        time.sleep(waitTime)
        wait = False
        car1.ResetPos()
        ag.ResetAgent()
    if pygame.mouse.get_pressed()[0]:
        ag.ResetAgent()
        ag.pos = pygame.Vector2(pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1])
        print(pygame.mouse.get_pos())
    dt = clock.tick(60) / 1000
    ct += dt
    screen.fill([120, 120, 110])
    keys = pygame.key.get_pressed()
    DrawTrack(track, False)
    car1.UpdateCar(keys)
    if not trainMode:
        if not wait and pygame.key.get_pressed()[pygame.K_r]:
            wait = True
        ag.TrainedMode()
    else:
        Agent.UpdateAgents()

    pygame.display.flip()

pygame.quit()
