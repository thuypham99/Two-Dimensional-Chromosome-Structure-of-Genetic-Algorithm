- 👋 Hi, I’m Thuy Pham Tran Minh
- 👀 This is the thesis topic for the Degree of Bachelor of Engineering in Logistics and Supply Chain Management
- 📫 How to reach me: phamtranminhthuy@gmail.com

✨ METAHEURISTIC FOR STAFF SCHEDULING PROBLEM IN ORDER FULFILLMENT: A CASE STUDY OF LAZADA ELOGISTICS VIETNAM ✨

- Objectives of Study: 

The solution of the scheduling model in this study will help not only Lazada E-Logistics Express but also other e-commerce platforms which have its order fulfillment service launch its outbound sales plan. When a vast majority of customer orders occurs, the outbound operation needs to work in full capacity. The rostering team needs some support tools to assign both temporary and official staff effectively in each area. Therefore, utilizing this rostering implementation in sales campaigns will help a company make the right decision at the right time to hire employees with affordable costs whilst satisfying worker preferences. These tools also can help managers reduce a bottleneck caused by improper assignment. Simultaneously, building a transparent contract including legal regulations, workplace policies, penalty, fixed labor cost, bonus cost, and other conditions not only ensure the rights and duties of employees but also help a warehouse avoid a sudden withdrawal.

- Scope:

The use of scheduling methods is crucial in building an effective real-world staff scheduling problem. This paper is aimed at solving a problem domain from order fulfillment industry, which plays a key vital to satisfy such a sudden increase of customer satisfactions, especially in Covid-19 pandemic by and large. There are two main objective functions including minimize the hiring cost and maximize the aspirations of workers. 

In this research, Requirement-Based Staff Scheduling Algorithm (RSSA) is introduced to compare with a two-phase Mixed Integer Programming and Genetic Algorithm with two-dimensional array chromosome structure. Mathematical model of phase 1 is implemented to give a fesible solution for the first target. Besides, experimental results highlight that RSSA and mathematical model of phase 2 could be applied effectively in current scale for the second goal. Especially, this novel algorithm tends to save more time whilst the Mixed Integer Programming model seems to satisfy high percentage of staff preferences when a demand forecasting is fluctuated. On the other hand, Genetic Algorithm is recommended in case of a scale of data is immense. 

------------- MODELLING FOR TWO-DIMENSIONAL CHROMOSOME STRUCTURE OF GENETIC ALGORITHM -------------

Check out the reference: J. S. Dean, “Staff Scheduling by a Genetic Algorithm with a Two-Dimensional Chromosome Structure,” p. 15.

Based on the analysis of  Dean, the structure of chromosome including a two-dimensional array is suitable for a kind of solutions having tables form such as scheduling working days for staff. The result of Dean's paper also proves that this chromosome approach converges quickly to reach either an optimal solution or a near-optimal solution than a traditional bit-string chromosome structure. Specifically, the array in this structure has a range of rows representing the maximum number of employees in rotation (offRotation) and a range of column representing the working days. 

![image](https://user-images.githubusercontent.com/88264932/127765550-0fa61607-d18e-49ef-b839-4d8a1a898a23.png)

The pseudo code of this GA is given below.

- Notations:

offWorkDaily: The number of employees needed each day ( e=1,…,E)

OffRegister: The request day-off of official employees

offRotation: The maximum number of employees to satisfy requirement.

day_no: The number of working days (d=1,…,D)

CrossProb: Probability of crossover operator

MutationProb: Probability of mutation operator

workingDayRequirement: The number of employees required for day d

![image](https://user-images.githubusercontent.com/88264932/127765633-bb080bab-1b0c-4c81-96aa-da74234de334.png)
![image](https://user-images.githubusercontent.com/88264932/127765643-655f7aa4-cdc8-4123-8acb-bb62f32d1090.png)

The general application of this GA is aimed at satisfying the maximum staffing need for each department and each shift. Based on phase 1 in this study, we have the number of employees need and it will be used to determine the maximum employees in job rotation by applying the formula introduced by Dean.

![image](https://user-images.githubusercontent.com/88264932/127765675-5dd3b2c3-0381-4046-8550-dc5cd7b5650a.png)

Then, we could calculate the maximum number of workers pool by transform previous formula to this:

![image](https://user-images.githubusercontent.com/88264932/127765685-0e1c129d-16a8-4c07-bf70-be350ed6d83d.png)

These two numbers are utilized later for creating chromosomes and population. The number of chromosomes relies on the number of combinations of maximum employees in rotation and employees working per day. For instance, due to the constraint of this paper, if we need the maximum number of official employees working per day is 10. Therefore, the maximum number of workers in the rotation pool equals up to 12 (the proportion of staff who work 6 out of 7 days is 6/7 = 0.8571). Then, the size of the initial population will be the combination of 12 choose 10 workers:

![image](https://user-images.githubusercontent.com/88264932/127765691-eca07c0a-eec2-4782-97c7-d815b9fdbab2.png)











