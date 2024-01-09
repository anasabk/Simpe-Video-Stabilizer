OBJECTS = obj/stabcv.o obj/main.o

SEARCH_PATH = include

ifdef DEBUG
  CFLAGS = -g -D WITH_DEBUG=1
else
  CFLAGS = -O3
endif

ifdef CUDA_STAB
  CFLAGS += -D CUDA_STAB=1
endif

CFLAGS += -I$(SEARCH_PATH) `pkg-config opencv --cflags --libs`

obj/%.o: src/%.cpp
	mkdir -p $(@D)
	g++ -c $< -o $@ $(CFLAGS)

build: $(OBJECTS)
	g++ $(OBJECTS) -o stabilizer $(CFLAGS)

run:
	./stabilizer res/im1.jpeg res/im2.jpeg

clean:
	rm -rf obj stabilizer