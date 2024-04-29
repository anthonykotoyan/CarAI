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
trainMode = True
randomStart = False
carImage = pygame.image.load('car.png').convert()
savedNN = [[8, 3, 3, 5], [
    [0.5770055787083826, 0.7879039670826151, 0.834457989784543, -0.8438181635149853, -0.4856071440313703,
     0.2362568439989638, -0.21690157077085834, 1.0, 0.09516814201440174, -0.8788350657175551, -0.138561653817355,
     0.5945678086278539, 0.39335154920334126, -0.4795208751517604, -0.748816420298675, 0.5759574145400743,
     -0.49821457569720984, -0.460826474160527, -0.4296393238803493, 0.038648705432990926, -0.10038873082772373,
     -0.37044232195692145, 0.2997082063761306, -0.4475709260923382],
    [0.6402105819868052, -0.02654935757010679, 0.6000096996225468, 0.12489931615672939, -1.0, -0.6650616949598509,
     0.325693397511463, 0.15565431662717844, -0.8071720444647439],
    [-0.4887173514134278, -0.9137389212303837, 0.5690704702876686, -1.0, 0.2925418364222091, 0.3140090216031861,
     0.15175826342867588, 0.6932645129264425, -0.4471572760891208, -0.26421827126187475, -1.0, 0.47675939618115215,
     0.09148334091912669, 0.7706235697048974, 0.8331012356958548]],
           [[-0.8365637988653842, -0.7747052429818927, 0.44534315325928164],
            [-0.03966384546168206, -0.2647014532500317, -0.9374899873558977],
            [-0.7601745047497269, -0.7344742451930395, 0.6924219214394122, 0.8148028965905475, -0.2598055337532207]]]

track = [[(319, 747), (405, 751), (529, 754), (640, 752), (768, 730), (906, 702), (994, 667), (1054, 612), (1095, 542),
          (1109, 469), (1100, 392), (1057, 321), (982, 267), (876, 251), (791, 275), (729, 311), (653, 343), (552, 351),
          (423, 312), (350, 240), (344, 167), (333, 110), (320, 76), (269, 43), (190, 23), (127, 36), (82, 72),
          (57, 112), (46, 196), (54, 258), (84, 368), (117, 398), (190, 435), (254, 452), (310, 453), (356, 449),
          (413, 446), (479, 438), (556, 432), (632, 436), (688, 464), (709, 496), (694, 531), (659, 563), (609, 588),
          (540, 597), (446, 599), (367, 597), (286, 599), (215, 624), (161, 658), (140, 694), (143, 729), (180, 750),
          (319, 747)],
         [(333, 710), (483, 710), (665, 690), (860, 636), (994, 540), (1048, 430), (996, 358), (921, 344), (794, 359),
          (482, 390), (327, 318), (278, 194), (226, 109), (169, 114), (129, 148), (128, 197), (177, 295), (297, 365),
          (421, 391), (522, 390), (627, 393), (730, 407), (778, 443), (794, 494), (776, 569), (701, 607), (580, 622),
          (478, 635), (380, 645), (288, 656), (236, 674), (232, 697), (265, 710), (333, 710)], (519, 738),
         [[(575, 685), (581, 770)], [(622, 675), (665, 768)], [(670, 655), (753, 764)], [(723, 650), (856, 741)],
          [(819, 627), (953, 716)], [(873, 582), (1025, 666)], [(918, 555), (1093, 610)], [(975, 499), (1142, 520)],
          [(1028, 443), (1129, 407)], [(1013, 385), (1074, 302)], [(950, 362), (964, 267)], [(887, 379), (841, 241)],
          [(815, 381), (741, 279)], [(725, 396), (637, 306)], [(604, 378), (571, 312)], [(527, 386), (521, 309)],
          [(455, 386), (479, 279)], [(400, 362), (434, 264)], [(337, 329), (381, 240)], [(291, 261), (385, 167)],
          [(268, 184), (357, 111)], [(233, 141), (295, 32)], [(224, 117), (257, 28)], [(214, 113), (227, 29)],
          [(198, 109), (194, 22)], [(190, 112), (166, 18)], [(176, 120), (139, 30)], [(162, 123), (103, 42)],
          [(153, 142), (72, 78)], [(155, 165), (49, 136)], [(141, 180), (41, 181)], [(144, 207), (41, 239)],
          [(161, 224), (64, 290)], [(176, 268), (67, 377)], [(204, 294), (148, 421)], [(222, 322), (198, 443)],
          [(321, 344), (275, 472)], [(363, 368), (347, 466)], [(400, 385), (398, 459)], [(439, 382), (454, 447)],
          [(518, 390), (516, 442)], [(593, 372), (561, 436)], [(643, 395), (601, 442)], [(695, 411), (639, 449)],
          [(737, 421), (659, 461)], [(776, 418), (675, 474)], [(785, 449), (702, 477)], [(754, 479), (679, 507)],
          [(758, 502), (673, 517)], [(750, 529), (650, 537)], [(722, 556), (659, 551)], [(709, 603), (632, 552)],
          [(650, 612), (605, 574)], [(575, 625), (547, 597)], [(505, 637), (477, 588)], [(462, 631), (431, 590)],
          [(420, 636), (389, 603)], [(384, 645), (337, 590)], [(344, 639), (274, 581)], [(300, 645), (234, 603)],
          [(264, 667), (169, 624)], [(242, 689), (118, 691)], [(252, 706), (217, 746)], [(273, 704), (262, 762)],
          [(326, 699), (319, 749)], [(372, 699), (371, 757)], [(428, 707), (432, 759)], [(479, 708), (485, 756)]]]


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
    genLength = 30
    startTime = ct
    boxSize = 2

    numBestAgents = 3
    mutateFactor = .2
    mutationChance = .25
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

    def CheckIfBad(self):
        if ct - Agent.startTime > Agent.genLength * .125:
            if self.fitness <= 1:
                self.AgentDeath()
        for i in range(3):
            if ct - Agent.startTime > Agent.genLength * (.25 * (i + 1)):
                if abs(self.speed) <= 1:
                    self.AgentDeath()

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
