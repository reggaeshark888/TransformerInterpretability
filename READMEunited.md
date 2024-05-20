### Table 1: Employees
| EmployeeID | Name         | Department   | Salary | JoinDate   |
|------------|--------------|--------------|--------|------------|
| 0          | John Smith   | Sales        | 60000  | 2022-01-15 |
| 1          | Jane Doe     | Marketing    | 70000  | 2022-02-20 |
| 2          | Mike Johnson | IT           | 80000  | 2022-03-10 |
| 3          | Emily Davis  | HR           | 60000  | 2022-04-25 |
| 4          | James Brown  | Sales        | 75000  | 2022-05-30 |

### Table 2: Bonuses
| BonusID | EmployeeID | BonusAmount | BonusDate   |
|---------|------------|-------------|-------------|
| 0       | 2          | 5000        | 2022-06-15  |
| 1       | 1          | 5000        | 2022-07-20  |
| 2       | 2          | 5000        | 2022-08-10  |
| 3       | 0          | 5000        | 2022-09-25  |
| 4       | 4          | 5000        | 2022-10-30  |

### Query Scenarios

#### Scenario 1: WHERE and Aggregation using SUM and MEAN
1. **Description**: Calculate the total and average salary for employees who joined after March 1, 2022.
2. **Query**:
    ```sql
    SELECT SUM(Salary) AS TotalSalary, AVG(Salary) AS AverageSalary
    FROM Employees
    WHERE JoinDate > '2022-03-01';
    ```

#### Scenario 2: JOIN two tables and perform an operation
1. **Description**: Calculate the total compensation (Salary + Bonus) for each employee.
2. **Query**:
    ```sql
    SELECT e.EmployeeID, e.Name, e.Salary, b.BonusAmount, (e.Salary + b.BonusAmount) AS TotalCompensation
    FROM Employees e
    JOIN Bonuses b ON e.EmployeeID = b.EmployeeID;
    ```

#### Scenario 3: GROUP BY
1. **Description**: Find the total salary paid per department.
2. **Query**:
    ```sql
    SELECT Department, SUM(Salary) AS TotalSalary
    FROM Employees
    GROUP BY Department;
    ```

#### Scenario 4: Equality Condition
1. **Description**: Retrieve employees who have the same salary.
2. **Query**:
    ```sql
    SELECT *
    FROM Employees
    WHERE Salary IN (SELECT Salary FROM Employees GROUP BY Salary HAVING COUNT(*) > 1);
    ```

#### Scenario 5: Aggregation by Month
1. **Description**: Calculate the total bonus given each month.
2. **Query**:
    ```sql
    SELECT DATE_FORMAT(BonusDate, '%Y-%m') AS Month, SUM(BonusAmount) AS TotalBonus
    FROM Bonuses
    GROUP BY DATE_FORMAT(BonusDate, '%Y-%m');
    ```
