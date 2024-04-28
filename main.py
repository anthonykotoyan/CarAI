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
dt = clock.tick(60) / 1000
trainMode = True
carImage = pygame.image.load('car.png').convert()
savedNN = [[6, 2, 5], [
    [-0.5631923790655183, 0.49841881290942175, 0.3465879773517755, -0.7998544662924352, -0.7634656680723565,
     -0.3329824442765934, 0.06965184167619747, 0.6615534093398112, -0.9207512985399936, 0.17195420942525624,
     -0.27753755545795084, 0.1654132749135642],
    [0.11100084913629474, -0.35000021188590874, -0.4665983482487088, -0.31395056736985283, 0.3223841505483032,
     0.4330940765418644, 0.1830718409702164, 0.7205411527589529, -0.9793688139628357, 0.21412091574162145]],
           [[-0.9363793491549001, -0.7692096667117796],
            [0.21072800178032325, -0.7209291988404987, -0.17757812180455015, 0.27404678192793813, -0.8020575054631593]]]
print(savedNN[1])
track = [[(852, 728), (881, 705), (905, 675), (928, 603), (931, 589), (932, 547), (897, 450), (857, 414), (765, 371),
          (632, 353), (473, 359), (345, 366), (280, 355), (221, 337), (226, 310), (236, 305), (325, 303), (439, 314),
          (579, 301), (729, 281), (825, 249), (889, 175), (863, 122), (687, 47), (468, 45), (262, 47), (98, 65),
          (54, 111), (59, 285), (51, 606), (55, 704), (335, 737), (852, 728)],
         [(288, 641), (549, 649), (620, 639), (701, 635), (734, 617), (771, 597), (763, 567), (760, 530), (741, 510),
          (647, 475), (523, 445), (419, 446), (259, 445), (159, 421), (156, 393), (149, 341), (173, 269), (222, 231),
          (353, 219), (610, 207), (722, 199), (763, 189), (765, 162), (723, 125), (643, 112), (468, 120), (221, 122),
          (107, 156), (131, 320), (139, 521), (177, 589), (211, 607), (288, 641)], (605, 686),
         [[(699, 597), (734, 745)], [(737, 594), (787, 743)], [(757, 592), (899, 696)], [(751, 571), (980, 567)],
          [(754, 533), (948, 517)], [(737, 520), (925, 473)], [(702, 513), (900, 439)], [(694, 501), (863, 407)],
          [(653, 489), (806, 382)], [(623, 476), (749, 350)], [(582, 467), (682, 345)], [(551, 465), (625, 336)],
          [(514, 461), (549, 347)], [(481, 454), (503, 355)], [(452, 457), (476, 356)], [(435, 448), (454, 353)],
          [(395, 459), (415, 351)], [(347, 447), (370, 355)], [(309, 451), (329, 362)], [(285, 441), (305, 361)],
          [(231, 441), (280, 339)], [(201, 432), (255, 340)], [(167, 427), (239, 335)], [(158, 393), (229, 333)],
          [(151, 364), (225, 322)], [(148, 326), (230, 313)], [(161, 294), (230, 311)], [(179, 267), (234, 306)],
          [(200, 248), (251, 305)], [(232, 234), (263, 304)], [(257, 229), (279, 308)], [(286, 225), (289, 302)],
          [(317, 224), (309, 299)], [(353, 222), (358, 309)], [(401, 219), (415, 315)], [(459, 214), (477, 304)],
          [(500, 215), (513, 302)], [(555, 213), (569, 301)], [(597, 216), (618, 287)], [(645, 209), (663, 280)],
          [(693, 219), (713, 285)], [(732, 197), (751, 271)], [(751, 199), (783, 260)], [(766, 178), (838, 235)],
          [(763, 163), (871, 200)], [(761, 158), (883, 165)], [(747, 147), (856, 121)], [(737, 136), (775, 89)],
          [(713, 124), (725, 75)], [(684, 113), (673, 52)], [(638, 109), (621, 52)], [(580, 112), (573, 51)],
          [(528, 117), (526, 48)], [(478, 119), (477, 52)], [(412, 115), (413, 47)], [(338, 116), (340, 57)],
          [(301, 117), (291, 57)], [(257, 118), (247, 60)], [(213, 123), (199, 65)], [(175, 135), (154, 65)],
          [(139, 143), (101, 67)], [(117, 153), (75, 95)], [(94, 146), (59, 113)], [(88, 171), (57, 170)],
          [(117, 213), (63, 220)], [(117, 250), (65, 263)], [(120, 291), (68, 304)], [(124, 350), (69, 354)],
          [(123, 385), (66, 389)], [(135, 437), (67, 435)], [(135, 475), (67, 472)], [(135, 508), (91, 511)],
          [(146, 533), (91, 573)], [(169, 573), (107, 657)], [(202, 601), (167, 695)], [(262, 628), (224, 709)],
          [(316, 641), (293, 719)], [(365, 638), (359, 723)], [(436, 653), (434, 720)], [(486, 649), (489, 723)],
          [(564, 653), (563, 726)], [(586, 646), (590, 723)]]]


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


layers = [len([-90, -37.5, 0, 37.5, 90]) + 1, 2, 5]


class Agent:
    maxSpeed = 6
    driftLength = 4
    maxTurnSpeed = 2.5
    driftFriction = 10
    friction = 0.4
    baseTSValue = -.3
    ShiftUpTS = 1.1
    driftSpeed = 1
    minDriftAngle = 20
    normalSlip = .2
    handBrakeSlip = normalSlip * 3
    errorCorrectionStrength = 15
    gen = 0
    genLength = 15
    startTime = time.time()
    boxSize = 2.5

    numBestAgents = 3
    mutateFactor = 0.01
    mutationChance = .01
    allAgents = []
    allMaxCP = 0
    nnBest = 0

    bestFitness = 0

    startAngle = 0

    def __init__(self):
        self.image = pygame.image.load('car.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 10, self.size * 10))
        self.acc = 3
        self.turnSpeed = 2.8
        self.slip = Agent.normalSlip
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
        self.angles = [-90, -37.5, 0, 37.5, 90]
        self.vision = []
        self.fitness = 0
        self.nextCP = 0
        self.runAgent = True
        randomTrade = False
        if randomTrade:
            self.nn = nn(layers)
            self.nn.randomize()
        else:
            self.nn = copyNN(savedNN)
            self.nn.mutate(Agent.mutationChance, Agent.mutateFactor)
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
        rotated_image = pygame.transform.rotate(self.image, -self.angle)
        # Update the car's rectangle to the new rotated image's rectangle
        self.car = rotated_image.get_rect(center=self.car.center)
        # Draw the rotated image onto the screen
        screen.blit(rotated_image, self.car)

    def DriftTrail(self, dist, wid):
        driftAngle = abs(((self.angle - self.dir + 540) % 360 - 180))
        if driftAngle >= Agent.minDriftAngle and abs(self.speed) > Agent.driftSpeed:
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

    def TrackCheckpoints(self):
        coll = BoxCollision(self.carCorners, track[3][self.nextCP])
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
        if abs(self.speed) >= Agent.maxSpeed:
            self.speed = Agent.maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, outputs):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if outputs[0]:
            self.speed -= self.acc * dt
        if abs(self.speed) > Agent.driftSpeed:
            self.speed += Agent.driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += Agent.friction * -sign(self.speed) * dt

        if outputs[4]:
            self.slip = Agent.handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        else:
            self.slip = Agent.normalSlip

        if outputs[1]:
            if self.speed < 0:
                self.speed += self.acc * dt
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
                self.dir += sign(angleError) * Agent.errorCorrectionStrength * (abs(angleError) / 180)


        elif outputs[3]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * Agent.errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (Agent.driftLength * .25)
        b = 4 * Agent.maxTurnSpeed / Agent.maxSpeed ** 2
        # self.turnSpeed = -b * (Agent.baseTSValue + abs(self.speed)) * (
        #         Agent.baseTSValue - Agent.maxSpeed + abs(self.speed)) + Agent.ShiftUpTS

    @staticmethod
    def RestartGen():

        fitnessMin = 1
        Agent.startTime = time.time()
        # a list of all agents fitness
        allFitness = [agent.fitness for agent in Agent.allAgents]

        # create a new list for the index of best fitness
        bestFitness = []
        for agents in range(Agent.numBestAgents):
            # some weird asss way to get index of nTh biggest number
            nThBestAgentIndex = sorted(range(len(allFitness)), key=lambda i: allFitness[i])[-(agents + 1)]
            bestFitness.append(nThBestAgentIndex)

        if max(allFitness) >= Agent.bestFitness:
            Agent.nnBest = Agent.allAgents[bestFitness[0]].nn
            Agent.bestFitness = max(allFitness)

            # loop through all agents to update nn
        for i, agent in enumerate(Agent.allAgents):
            print(agent.nn.weights)
            if max(allFitness) > fitnessMin:

                agent.nn = Agent.nnBest
                agent.nn.mutate(Agent.mutationChance, Agent.mutateFactor)
                agent.ResetAgent()
            else:

                agent.nn = nn(layers)
                agent.nn.randomize()
                agent.ResetAgent()
                print(max(allFitness))


    @staticmethod
    def ManageGen():

        if time.time() - Agent.startTime > Agent.genLength:
            Agent.RestartGen()
            Agent.gen += 1
            Agent.startTime = time.time()

    def RunAgent(self):
        nnInputs = self.vision[0]
        nnInputs.append(self.speed)
        nnOutputs = self.nn.run(nnInputs, nn.Tanh)
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
            la_ = Agent.nnBest.layers
            we_ = Agent.nnBest.weights
            ba_ = Agent.nnBest.biases
            print([la_, we_, ba_])
        for i in range(len(Agent.allAgents)):
            if Agent.allAgents[i].runAgent:
                Agent.allAgents[i].ApplyVelocity()
                Agent.allAgents[i].ApplyDirection()
                Agent.allAgents[i].RunAgent()

                Agent.allAgents[i].TrackCheckpoints()
                Agent.allAgents[i].TrackCollisions()
                Agent.allAgents[i].BorderCollisions()

                Agent.allAgents[i].DriftTrail(10 * Agent.allAgents[i].size, 4 * Agent.allAgents[i].size)
                Agent.allAgents[i].DrawAngle(50, False)
                Agent.allAgents[i].DrawCar()

            else:
                deadAgents += 1
        if deadAgents >= len(Agent.allAgents):
            Agent.RestartGen()

        allFitness = [agent.fitness for agent in Agent.allAgents]
        if max(allFitness) >= Agent.bestFitness:
            Agent.nnBest = Agent.allAgents[max(allFitness)].nn
            Agent.bestFitness = max(allFitness)
        Agent.allAgents[allFitness.index(max(allFitness))].HitBox(True)
        Agent.allAgents[allFitness.index(max(allFitness))].DrawVision(
            Agent.allAgents[allFitness.index(max(allFitness))].vision[1])

    def TrainedMode(self):
        self.ApplyVelocity()
        self.ApplyDirection()
        self.RunAgent()
        self.DrawVision(self.vision[1])
        self.TrackCheckpoints()
        self.TrackCollisions()
        self.BorderCollisions()

        self.DriftTrail(10 * self.size, 4 * self.size)
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
    ag.nn = copyNN(savedNN)  # put a good nn


class Car:
    maxSpeed = 6
    driftLength = 4
    maxTurnSpeed = 2.5
    driftFriction = 10
    friction = 0.4
    baseTSValue = -.3
    ShiftUpTS = 1.1
    driftSpeed = 1
    minDriftAngle = 20
    normalSlip = .2
    handBrakeSlip = normalSlip * 3
    errorCorrectionStrength = 15

    boxSize = 2.5

    def __init__(self):
        self.image = pygame.image.load('car.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 10, self.size * 10))
        self.acc = 3
        self.turnSpeed = 2.8
        self.slip = Car.normalSlip
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
        if driftAngle >= Car.minDriftAngle and abs(self.speed) > Car.driftSpeed:
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
        if abs(self.speed) >= Car.maxSpeed:
            self.speed = Car.maxSpeed * sign(self.speed)
        self.vel.x = -math.cos(math.radians(self.dir)) * self.speed
        self.vel.y = -math.sin(math.radians(self.dir)) * self.speed

    def Controls(self, keys):
        angleError = ((self.angle - self.dir + 540) % 360 - 180)
        if keys[pygame.K_w]:
            self.speed -= self.acc * dt
        if abs(self.speed) > Car.driftSpeed:
            self.speed += Car.driftFriction * dt * (abs(angleError) / 180) * -sign(self.speed)
        self.speed += Car.friction * -sign(self.speed) * dt
        if keys[pygame.K_SPACE]:
            self.slip = Car.handBrakeSlip
            if self.speed < 0:
                self.speed += self.acc * dt
            else:
                self.speed = 0
        else:
            self.slip = Car.normalSlip

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
                self.dir += sign(angleError) * Car.errorCorrectionStrength * (abs(angleError) / 180)


        elif keys[pygame.K_a]:
            # angle error changes negatively
            self.angle -= self.turnSpeed
            if self.speed != 0:
                self.dir -= self.turnSpeed * (1 - self.slip)
            else:
                self.dir = self.angle
            if angleError > 0:
                self.dir += sign(angleError) * Car.errorCorrectionStrength * (abs(angleError) / 180)


        else:
            self.dir += sign(angleError) * (Car.driftLength * .25)
        b = 4 * Car.maxTurnSpeed / Car.maxSpeed ** 2
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

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill("grey")
    keys = pygame.key.get_pressed()
    DrawTrack(track, False)
    car1.UpdateCar(keys)
    if trainMode:
        Agent.UpdateAgents()
    else:
        ag.TrainedMode()

    pygame.display.flip()
    dt = clock.tick(60) / 1000

pygame.quit()
