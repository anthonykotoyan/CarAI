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
carImage = pygame.image.load('car.png').convert()
savedNN = [[8, 3, 3, 5], [
    [-0.6388361750360847, -0.48924881198264936, -0.7229851326598407, -0.41431195299849577, 0.9607785027725644,
     0.29534095165563046, 0.7514586501224254, 0.05146282938927954, 1.0, 0.5416875995087731, 0.7128895887813841,
     0.7734736401192841, 0.34323059238705905, 0.9655134243088321, -0.5894313318420121, -0.28362212546778354,
     -0.43778487003615474, -1.0, -0.19282984486727528, 0.45707500744051444, -0.4623491055410939, -0.7828416471419446,
     -0.4119842317283256, 1.0],
    [-1.0, -0.37213924673645565, -0.21738689999490768, 0.016482685529670554, 0.6670472063988104, 0.041425497287444324,
     0.32562260611952804, 0.7391009035309655, 1.0],
    [-0.6414847520045611, 0.18804454097920692, 0.20153306759810574, -0.8386089168177993, -1.0, 0.7511218771719927,
     -0.0767042199148707, -1.0, -0.5239222184783732, 0.1897725786594076, -0.5928275776746197, 0.48479389608641643,
     -0.3643672254272507, 1.0, 0.09699088814902806]], [[0.9543376855385516, -0.8164110259273892, 0.12116440816815607],
                                                       [1.0, -0.5618609920860278, 0.6538132818878131],
                                                       [1.0, -1.0, 0.04147366494557092, 0.17855045064028785,
                                                        -0.18115753396073275]]]

track = [[(206, 737), (253, 751), (340, 755), (427, 758), (538, 743), (629, 714), (704, 674), (769, 632), (846, 601),
          (912, 604), (975, 633), (1052, 676), (1141, 686), (1207, 651), (1251, 583), (1280, 471), (1273, 355),
          (1252, 262), (1201, 161), (1134, 108), (1020, 80), (922, 97), (809, 156), (704, 208), (589, 215), (486, 143),
          (383, 57), (247, 27), (106, 94), (50, 211), (36, 401), (61, 577), (122, 729), (206, 737)],
         [(236, 599), (297, 624), (432, 605), (539, 565), (648, 522), (741, 503), (846, 501), (936, 516), (1048, 526),
          (1120, 482), (1163, 379), (1160, 309), (1108, 273), (1050, 265), (937, 293), (824, 337), (672, 359),
          (540, 324), (465, 256), (377, 176), (247, 154), (165, 214), (145, 311), (159, 422), (200, 549), (236, 599)],
         (382, 715),
         [[(457, 556), (563, 760)], [(809, 458), (865, 654)], [(1099, 453), (1279, 558)], [(1092, 305), (1241, 137)],
          [(878, 319), (807, 132)], [(413, 315), (519, 92)], [(225, 195), (119, 52)], [(192, 386), (-1, 442)],
          [(251, 486), (260, 771)]]]


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


layers = [len([-90, -37.5, -15, 0, 15, 37.5, 90]) + 1, 3, 3, 5]

acceleration = 2
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
normalSlip = .2
handBrakeSlip = normalSlip * 1.5
errorCorrectionStrength = 15


class Agent:
    gen = 0
    genLength = 25
    startTime = ct
    boxSize = 2.5

    numBestAgents = 3
    mutateFactor = .4
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
        randomStart = True
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
