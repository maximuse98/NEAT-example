import pygame
import neat
import math
import random
import pickle
import time
import os

WIN_WIDTH = 600
WIN_HEIGHT = 600

ROAD_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "road.png")))
MAIN_CAR_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "main_car2.png")))
CAR_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "car3.png")))

ROAD_SIZE = 3
SPEED = 20

pygame.font.init()
STAT_FONT = pygame.font.SysFont("comicsans", 50)

GEN = 0
WIN = False

class MainCar:
	IMG = MAIN_CAR_IMG

	def __init__(self, position):
		self.position = position
		self.x = 155 + self.position*110
		self.y = 430
		self.img = self.IMG

	def turn(self, isLeft):
		if isLeft == True and position >= 0:
			position -= 1
		if isLeft == False and position <= 2:
			position += 1

	def draw(self, win):
		win.blit(self.IMG, (self.x, self.y))

	def get_mask(self):
		return pygame.mask.from_surface(self.img)

	def setPosition(self, position):
		self.position = position
		self.x = 155 + self.position*110

	def getPosition(self):
		return self.position

	def turnLeft(self):
		if self.position != 0:
			self.x = self.x - 100
			self.position -= 1

	def turnRight(self):
		if self.position != ROAD_SIZE-1:
			self.x = self.x + 100
			self.position += 1

class Car:
	IMG = CAR_IMG
	START_Y = -200
	END_Y = 600

	def __init__(self, position):
		self.position = position
		self.x = 120 + self.position*100
		self.y = self.START_Y
		self.passed = False
		self.img = self.IMG

	def move(self):
		self.y += SPEED

	def draw(self, win):
		win.blit(self.IMG, (self.x, self.y))

	def getPosition(self):
		return self.position

	def collide(self, mainCar):
		mainCar_mask = mainCar.get_mask()
		car_mask = pygame.mask.from_surface(self.img)

		offset = (self.x - mainCar.x, self.y - round(mainCar.y))

		check_point = mainCar_mask.overlap(car_mask, offset)

		if check_point:
			return True

		return False

class Base:
	HEIGHT = ROAD_IMG.get_height()
	IMG = ROAD_IMG

	def __init__(self):
		self.x = 0
		self.y1 = 0
		self.y2 = self.HEIGHT

	def move(self):
		self.y1 += SPEED
		self.y2 += SPEED

		if self.y1 - self.HEIGHT > 0:
			self.y1 = self.y2 - self.HEIGHT

		if self.y2 - self.HEIGHT > 0:
			self.y2 = self.y1 - self.HEIGHT

	def draw(self, win):
		win.blit(self.IMG, (self.x, self.y1))
		win.blit(self.IMG, (self.x, self.y2))


def draw_window(win, base, mainCars, cars, score, generation_num, population_size):
	#win.blit(ROAD_IMG, (0,0))
	base.draw(win)

	for mainCar in mainCars:
		mainCar.draw(win)

	for car in cars:
		car.draw(win)

	text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
	win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))

	text = STAT_FONT.render("Gen: " + str(generation_num), 1, (255,255,255))
	win.blit(text, (10, 10))

	text = STAT_FONT.render("Population: " + str(population_size), 1, (255,255,255))
	win.blit(text, (10, 45))

	pygame.display.update()

def save_winner(winner):
	with open('winner.pickle', 'wb') as handle:
		pickle.dump(winner, handle, protocol = pickle.HIGHEST_PROTOCOL)
	print("---------------------------------------------------------------")
	print("Done! Network parametres is saved to 'winner.pickle'")

def main(genomes, config):
	win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
	pygame.display.flip()

	global WIN
	global GEN
	GEN += 1

	base = Base()
	mainCars = []
	cars = []
	nets = []
	ge = []

	for x in range(0, ROAD_SIZE):
		cars.append(Car(x))

	for _, g in genomes:
		net = neat.nn.FeedForwardNetwork.create(g, config)
		nets.append(net)
		mainCars.append(MainCar(1))
		g.fitness = 0
		ge.append(g)

	clock = pygame.time.Clock()
	score = 0

	run = True
	while run:
		clock.tick(30)
		base.move()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				run = False
				pygame.quit()
				quit()

		inputs = [0] * ROAD_SIZE

		for car in cars:

			for x,mainCar in enumerate(mainCars):
				if car.collide(mainCar):
					ge[x].fitness -= 1
					mainCars.pop(x)
					nets.pop(x)
					ge.pop(x)

			car.move()
			isCreate = False
			askGenetic = False
			if car.y == car.END_Y:
				cars.remove(car)
				score += 1
				askGenetic = True

				for g in ge:
					g.fitness += 5

		if len(cars) == 0:
			askGenetic = True
			positions = list(range(ROAD_SIZE))
			for position in positions:
				isCreate = bool(random.getrandbits(1))

				if isCreate == True:
					newCar = Car(position)
					cars.append(newCar)

			if len(cars) == 0:
				new_car_num = random.randrange(0, ROAD_SIZE-1)
				newCar = Car(new_car_num)
				cars.append(newCar)

		if len(cars) == ROAD_SIZE:
			askGenetic = True
			removed_car_num = random.randrange(0, ROAD_SIZE-1)
			del cars[removed_car_num]

		if len(mainCars) == 0:
			run = False
			break

		for car in cars:
			inputs[car.getPosition()] = 1

		if askGenetic == True:
			for x, mainCar in enumerate(mainCars):
				ge[x].fitness += 0.1

				outputs = nets[x].activate(inputs)

				for x1 in range(0, ROAD_SIZE):
					if max(outputs) == outputs[x1]:
						mainCar.setPosition(x1)

		if score >= 200:
			WIN = True
			break

		draw_window(win, base, mainCars, cars, score, GEN, len(ge))

def run(config_path):
	# Load configuration.
	config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

	# Create the population, which is the top-level object for a NEAT run.
	p = neat.Population(config)

	# Add a stdout reporter to show progress in the terminal.
	p.add_reporter(neat.StdOutReporter(True))
	stats = neat.StatisticsReporter()
	p.add_reporter(stats)

	# Run for up to N generations.
	winner = p.run(main,20)
	pygame.quit()

	if WIN == True:
		save_winner(winner)
	else:
		print("---------------------------------------------------------------")
		print("Winning generation did not found. Change parameters or try again")

if __name__ == "__main__":
	# Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
	local_dir = os.path.dirname(__file__)
	config_path = os.path.join(local_dir, "config-feedforward.txt")
	run(config_path)
