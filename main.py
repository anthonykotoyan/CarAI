import math
import random

from nn import NeuralNetwork as nn
import pygame

pygame.init()
width, length = 1300, 800
screen = pygame.display.set_mode((width, length))
clock = pygame.time.Clock()
running = True
ct = 0
dt = clock.tick(60) / 1000
trainMode = True
carImage = pygame.image.load('car.png').convert()
savedNN = [[6, 5], [
    [0.6284390156215718, -0.48829054413756057, 0.1420048941707951, 1.0, -0.26761139493081787, 1.0, 0.3667238541328865,
     -0.8075718351350559, 0.3170984477804146, -0.8940114592123334, 1.0, -1.0, -0.044498232835781415, 1.0, -1.0,
     -0.144223328844308, 0.7716561517414542, 0.4376930023949523, 0.503498867138263, -1.0, 0.07397045856059453, -1.0,
     -0.21266038531863007, 0.7973763400994281, -0.760273538674815, -0.29406680172391597, 0.0935434468046942,
     -0.2890765827977021, -0.5626495400557969, 0.3990297960624647]],
           [[0.8286433441063557, 0.6856119444291343, -0.17434540962905476, -0.8276435221397287, -0.35428503913205844]]]

track = [
    [(515, 735), (734, 754), (921, 742), (1064, 678), (1137, 565), (1159, 479), (1153, 416), (1135, 300), (1097, 237),
     (1034, 216), (966, 210), (928, 216), (892, 227), (843, 234), (805, 229), (784, 205), (762, 161), (724, 120),
     (631, 67), (531, 41), (389, 31), (263, 37), (148, 65), (86, 146), (56, 259), (81, 324), (122, 339), (194, 351),
     (278, 387), (296, 432), (286, 487), (260, 546), (242, 622), (261, 672), (305, 705), (378, 721), (459, 725),
     (515, 735)],
    [(624, 636), (770, 641), (883, 623), (972, 563), (1019, 496), (1019, 436), (971, 375), (816, 336), (705, 314),
     (669, 268), (653, 220), (570, 153), (389, 107), (243, 125), (191, 194), (217, 248), (321, 293), (389, 356),
     (394, 423), (380, 526), (403, 595), (494, 619), (624, 636)], (872, 673),
    [[(894, 581), (1035, 715)], [(939, 547), (1130, 639)], [(973, 510), (1186, 538)], [(980, 448), (1187, 426)],
     [(977, 407), (1182, 310)], [(925, 385), (1064, 162)], [(866, 369), (914, 167)], [(767, 356), (844, 185)],
     [(678, 322), (812, 168)], [(622, 244), (767, 94)], [(579, 202), (658, 67)], [(509, 173), (563, 27)],
     [(441, 174), (465, 8)], [(390, 140), (338, 14)], [(333, 153), (210, 23)], [(256, 147), (100, 95)],
     [(248, 175), (67, 201)], [(232, 224), (60, 321)], [(277, 230), (206, 427)], [(383, 293), (243, 442)],
     [(477, 421), (260, 508)], [(442, 524), (142, 632)], [(432, 576), (291, 720)], [(485, 573), (459, 748)],
     [(544, 577), (555, 766)], [(631, 613), (669, 773)], [(733, 608), (802, 764)], [(817, 614), (883, 765)]]]


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


layers = [len([-90, -37.5, 0, 37.5, 90]) + 1, 5]

maxSpeed = 6
driftLength = 10
driftLength = 4
turnSpeed = 3.5
driftFriction = 13
friction = 0.4
baseTSValue = -.3
ShiftUpTS = 1.1
driftSpeed = .7
minDriftAngle = 20
normalSlip = .2
handBrakeSlip = normalSlip * 1.5
errorCorrectionStrength = 15


class Agent:
    gen = 0
    genLength = 20
    startTime = ct
    boxSize = 2.5

    numBestAgents = 3
    mutateFactor = .4
    mutationChance = .3
    allAgents = []
    allMaxCP = 0
    nnBest = 0

    bestFitness = 0

    startAngle = 0

    def __init__(self):
        self.image = pygame.image.load('babyman.png').convert_alpha()
        self.size = 3
        self.image = pygame.transform.scale(self.image, (self.size * 15, self.size * 15))
        self.acc = 3
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
        self.angles = [-90, -37.5, 0, 37.5, 90]
        self.vision = []
        self.fitness = 0
        self.nextCP = 0
        self.runAgent = True
        randomStart = False
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
                self.speed += self.acc * dt
            else:
                self.speed = 0
        else:
            self.slip = normalSlip

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

    @staticmethod
    def ManageGen():

        if ct - Agent.startTime > Agent.genLength:
            Agent.RestartGen()
            Agent.gen += 1
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

                agent.TrackCheckpoints(True)
                agent.TrackCollisions()
                agent.BorderCollisions()

                agent.DriftTrail(10 * agent.size, 5 * agent.size)
                agent.DrawAngle(50, False)
                agent.DrawCar()

            else:

                deadAgents += 1
        if deadAgents >= len(Agent.allAgents):
            Agent.gen += 1
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


numAgents = 40
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
        self.acc = 3
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
                self.speed += self.acc * dt
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

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    dt = clock.tick(60) / 1000
    ct += dt
    screen.fill([120, 120, 110])
    keys = pygame.key.get_pressed()
    DrawTrack(track, False)
    car1.UpdateCar(keys)
    if trainMode:
        Agent.UpdateAgents()
    else:
        ag.TrainedMode()

    pygame.display.flip()

pygame.quit()
