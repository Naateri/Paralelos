//Based from http://www.qnx.com/developers/docs/qnxcar2/index.jsp?topic=%2Fcom.qnx.doc.neutrino.getting_started%2Ftopic%2Fs1_procs_barriers.html

//pthread barriers
//Compile: gcc pth_bar.c -o pth_bar -lpthread

#include <stdio.h>
#include <time.h>
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>

pthread_barrier_t   barrier; // the barrier synchronization object

void *
thread1 (void *not_used)
{
    time_t  now;
    char    buf [27];

    time (&now);
    printf ("thread1 starting at %s", ctime_r (&now, buf));

    // do the computation
    // let's just do a sleep here...
    sleep (5);
    pthread_barrier_wait (&barrier);
    // after this point, all three threads have completed.
    time (&now);
    printf ("barrier in thread1() done at %s", ctime_r (&now, buf));
}

void *
thread2 (void *not_used)
{
    time_t  now;
    char    buf [27];

    time (&now);
    printf ("thread2 starting at %s", ctime_r (&now, buf));

    // do the computation
    // let's just do a sleep here...
    sleep (7);
    pthread_barrier_wait (&barrier);
    // after this point, all three threads have completed.
    time (&now);
    printf ("barrier in thread2() done at %s", ctime_r (&now, buf));
}

int main () // ignore arguments
{
    time_t  now;
    char    buf [27];
	long    thread;  /* Use long in case of a 64-bit system */
	pthread_t* thread_handles; 
	int thread_count = 2;

    // create a barrier object with a count of 3
    pthread_barrier_init (&barrier, NULL, 3);

	thread_handles = malloc (thread_count*sizeof(pthread_t));

    // start up two threads, thread1 and thread2
    pthread_create (&thread_handles[0], NULL, thread1, NULL);
    pthread_create (&thread_handles[0], NULL, thread2, NULL);

    // at this point, thread1 and thread2 are running

    // now wait for completion
    time (&now);
    printf ("main () waiting for barrier at %s", ctime_r (&now, buf));
    pthread_barrier_wait (&barrier);

    // after this point, all three threads have completed.
    time (&now);
    printf ("barrier in main () done at %s", ctime_r (&now, buf));
	
	free(thread_handles);
	return 0;
}
